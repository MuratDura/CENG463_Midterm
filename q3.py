"""
Dimensionality reduction study (PCA, Kernel PCA, t-SNE, UMAP, autoencoder).

Dataset: Fashion-MNIST (OpenML) by default; set USE_FASHION_MNIST=False for MNIST digits.

Metrics:
  - Reconstruction MSE (test): PCA, Kernel PCA (inverse_transform), undercomplete autoencoder.
  - Trustworthiness + continuity (dual of trustworthiness): t-SNE and UMAP embeddings.
  - Kruskal Stress-1 on a pairwise-distance subsample for t-SNE/UMAP.
  - k-NN (k=5) with 5-fold stratified CV on reduced training coordinates.
  - Nested CV for PCA / Kernel PCA / AE (refit reducer each fold). For t-SNE/UMAP a single
    fit on training data is used (standard practice; full nested DR is prohibitive).

Requires: pip install umap-learn

Use FAST_MODE=True for quick runs; set False for larger samples and longer runs.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import mean_squared_error, pairwise_distances
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration ---
RANDOM_STATE = 42
FIG_DIR = Path(__file__).resolve().parent / "q3_figures"
RESULTS_CSV = Path(__file__).resolve().parent / "q3_metrics_summary.csv"

USE_FASHION_MNIST = True
FAST_MODE = True

# Subsample for DR / CV (stratified)
FAST_N_TRAIN = 8_000
FAST_N_TEST = 2_000
FULL_N_TRAIN = 55_000
FULL_N_TEST = 5_000

LATENT_DIM = 2
TSNE_PERPLEXITIES = (5, 30, 50)
UMAP_GRID = (
    (5, 0.0),
    (15, 0.1),
    (30, 0.25),
    (50, 0.1),
)

# Default AE widths (bottleneck set per-run via latent_dim in fit_evaluate_pca_kpca_ae).
AE_DECODER_WIDTHS = (128, 256)

NEIGHBORS_QUALITY = 15
STRESS_SUBSAMPLE = 4_000
KNN_K = 5
CV_SPLITS = 5

np.random.seed(RANDOM_STATE)


def continuity(
    X: np.ndarray,
    X_embedded: np.ndarray,
    *,
    n_neighbors: int = 15,
    metric: str = "euclidean",
) -> float:
    """
    Continuity (Venna & Kaski): dual of trustworthiness — kNN in input space, ranks in embedding.
    """
    n_samples = X.shape[0]
    if n_neighbors >= n_samples / 2:
        raise ValueError("n_neighbors must be < n_samples / 2")

    dist_embedded = pairwise_distances(X_embedded, metric=metric)
    np.fill_diagonal(dist_embedded, np.inf)
    ind_embedded = np.argsort(dist_embedded, axis=1)

    inverted_index = np.zeros((n_samples, n_samples), dtype=np.int32)
    ordered_indices = np.arange(n_samples + 1)
    inverted_index[ordered_indices[:-1, np.newaxis], ind_embedded] = ordered_indices[1:]

    ind_high = (
        NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        .fit(X)
        .kneighbors(return_distance=False)
    )

    ranks = inverted_index[ordered_indices[:-1, np.newaxis], ind_high] - n_neighbors
    t = float(np.sum(ranks[ranks > 0]))
    t = 1.0 - t * (
        2.0 / (n_samples * n_neighbors * (2.0 * n_samples - 3.0 * n_neighbors - 1.0))
    )
    return t


def kruskal_stress_1(
    X_high: np.ndarray,
    Z_low: np.ndarray,
    rng: np.random.Generator,
    max_pairs_sample: int = 4000,
) -> float:
    """
    Kruskal Stress-1 between high-D Euclidean distances and low-D Euclidean distances,
    with optimal isotropic scaling of low-D distances (least squares).
    Computed on a random subset of points (pairwise uses O(m^2) on subset size m).
    """
    n = X_high.shape[0]
    m = min(max_pairs_sample, n)
    idx = rng.choice(n, size=m, replace=False)
    Xs = X_high[idx]
    Zs = Z_low[idx]

    dh = pairwise_distances(Xs, metric="euclidean")
    dl = pairwise_distances(Zs, metric="euclidean")
    mask = np.triu(np.ones((m, m), dtype=bool), k=1)
    u = dh[mask]
    v = dl[mask]
    # Optimal scaling: alpha = sum(u*v) / sum(v^2)
    alpha = float(np.dot(u, v) / (np.dot(v, v) + 1e-12))
    resid = u - alpha * v
    stress = float(np.sqrt(np.sum(resid**2) / (np.sum(u**2) + 1e-12)))
    return stress


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def mlp_encode_bottleneck(
    mlp: MLPRegressor, X: np.ndarray, n_encoder_layers: int
) -> np.ndarray:
    """Forward pass through encoder layers only (up to bottleneck activations)."""
    a = X
    for i in range(n_encoder_layers):
        z = a @ mlp.coefs_[i] + mlp.intercepts_[i]
        a = _relu(z)
    return a


def rough_stroke_weight(images: np.ndarray) -> np.ndarray:
    """Simple proxy for ink / thickness: mean intensity per image."""
    return images.reshape(len(images), -1).mean(axis=1)


def load_mnist_like() -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    did = 40996 if USE_FASHION_MNIST else 554
    bunch = fetch_openml(data_id=did, as_frame=False, parser="auto")
    X = np.asarray(bunch.data, dtype=np.float32)
    y = np.asarray(bunch.target, dtype=np.int64)
    if y.min() >= 1:
        y = y - y.min()
    tnames = getattr(bunch, "target_names", None)
    class_names = list(tnames) if tnames is not None else None
    return X, y, class_names


def stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Balanced take per class then fill remainder."""
    classes = np.unique(y)
    per = n // len(classes)
    idx_list = []
    for c in classes:
        ic = np.where(y == c)[0]
        rng.shuffle(ic)
        idx_list.append(ic[:per])
    idx = np.concatenate(idx_list)
    if len(idx) < n:
        rest = np.setdiff1d(np.arange(len(y)), idx)
        rng.shuffle(rest)
        idx = np.concatenate([idx, rest[: n - len(idx)]])
    rng.shuffle(idx)
    return X[idx], y[idx]


def fit_evaluate_pca_kpca_ae(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    latent_dim: int,
) -> dict[str, Any]:
    """PCA, Kernel PCA, MLP AE: nested CV for kNN + test reconstruction MSE."""
    results: dict[str, Any] = {}

    # --- PCA ---
    t0 = time.perf_counter()
    pipe_pca = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=latent_dim, random_state=RANDOM_STATE)),
        ]
    )
    pipe_pca.fit(X_train)
    mem_pca = X_train.nbytes + X_test.nbytes  # rough footprint for discussion

    Z_train_pca = pipe_pca.transform(X_train)
    Z_test_pca = pipe_pca.transform(X_test)
    X_hat_test_pca = pipe_pca.inverse_transform(Z_test_pca)
    mse_pca = mean_squared_error(X_test, X_hat_test_pca)

    cv_pca = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores_pca = []
    for tr, va in cv_pca.split(X_train, y_train):
        p = Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=latent_dim, random_state=RANDOM_STATE)),
            ]
        )
        p.fit(X_train[tr])
        z_tr = p.transform(X_train[tr])
        z_va = p.transform(X_train[va])
        knn = KNeighborsClassifier(n_neighbors=KNN_K)
        knn.fit(z_tr, y_train[tr])
        scores_pca.append(knn.score(z_va, y_train[va]))
    t_pca = time.perf_counter() - t0

    results["pca"] = {
        "test_recon_mse": mse_pca,
        "knn_cv_acc_mean": float(np.mean(scores_pca)),
        "knn_cv_acc_std": float(np.std(scores_pca)),
        "fit_time_s": t_pca,
        "memory_note": f"~{mem_pca / 1e6:.1f} MB arrays (order-of-magnitude)",
        "Z_train": Z_train_pca,
        "Z_test": Z_test_pca,
        "name": "PCA",
    }

    # --- Kernel PCA (RBF) ---
    t0 = time.perf_counter()
    kpca = KernelPCA(
        n_components=latent_dim,
        kernel="rbf",
        gamma=None,
        fit_inverse_transform=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    scaler_k = StandardScaler()
    X_train_s = scaler_k.fit_transform(X_train)
    X_test_s = scaler_k.transform(X_test)
    kpca.fit(X_train_s)
    Z_train_k = kpca.transform(X_train_s)
    Z_test_k = kpca.transform(X_test_s)
    X_hat_test_k = kpca.inverse_transform(Z_test_k)
    mse_k = mean_squared_error(X_test_s, X_hat_test_k)  # inverse_transform lives in scaled input space
    # also report raw pixel MSE for interpretability: invert scaler on reconstruction
    X_hat_raw = scaler_k.inverse_transform(X_hat_test_k)
    mse_k_raw = mean_squared_error(X_test, X_hat_raw)

    cv_k = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores_k = []
    for tr, va in cv_k.split(X_train, y_train):
        sc = StandardScaler()
        Xt = sc.fit_transform(X_train[tr])
        Xv = sc.transform(X_train[va])
        k = KernelPCA(
            n_components=latent_dim,
            kernel="rbf",
            fit_inverse_transform=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        k.fit(Xt)
        knn = KNeighborsClassifier(n_neighbors=KNN_K)
        knn.fit(k.transform(Xt), y_train[tr])
        scores_k.append(knn.score(k.transform(Xv), y_train[va]))
    t_k = time.perf_counter() - t0

    results["kpca"] = {
        "test_recon_mse_scaled": mse_k,
        "test_recon_mse_pixels": mse_k_raw,
        "knn_cv_acc_mean": float(np.mean(scores_k)),
        "knn_cv_acc_std": float(np.std(scores_k)),
        "fit_time_s": t_k,
        "Z_train": Z_train_k,
        "Z_test": Z_test_k,
        "scaler": scaler_k,
        "kpca_model": kpca,
        "name": "Kernel PCA (RBF)",
    }

    # --- Undercomplete autoencoder (MLPRegressor): 3 encoder hiddens incl. bottleneck ---
    encoder_layers = (256, 128, latent_dim)
    hidden = (*encoder_layers, *AE_DECODER_WIDTHS)
    n_enc_layers = len(encoder_layers)
    t0 = time.perf_counter()
    scaler_ae = MinMaxScaler()
    X_train_n = scaler_ae.fit_transform(X_train)
    X_test_n = scaler_ae.transform(X_test)

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=400 if FAST_MODE else 800,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    mlp.fit(X_train_n, X_train_n)
    X_hat_te = mlp.predict(X_test_n)
    mse_ae = mean_squared_error(X_test_n, X_hat_te)

    Z_train_ae = mlp_encode_bottleneck(mlp, X_train_n, n_enc_layers)
    Z_test_ae = mlp_encode_bottleneck(mlp, X_test_n, n_enc_layers)

    cv_ae = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores_ae = []
    for tr, va in cv_ae.split(X_train, y_train):
        sc = MinMaxScaler()
        Xt = sc.fit_transform(X_train[tr])
        Xv = sc.transform(X_train[va])
        m = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            max_iter=300 if FAST_MODE else 600,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=RANDOM_STATE,
            verbose=False,
        )
        m.fit(Xt, Xt)
        z_tr = mlp_encode_bottleneck(m, Xt, n_enc_layers)
        z_va = mlp_encode_bottleneck(m, Xv, n_enc_layers)
        knn = KNeighborsClassifier(n_neighbors=KNN_K)
        knn.fit(z_tr, y_train[tr])
        scores_ae.append(knn.score(z_va, y_train[va]))
    t_ae = time.perf_counter() - t0

    results["ae"] = {
        "test_recon_mse": mse_ae,
        "knn_cv_acc_mean": float(np.mean(scores_ae)),
        "knn_cv_acc_std": float(np.std(scores_ae)),
        "fit_time_s": t_ae,
        "mlp": mlp,
        "scaler": scaler_ae,
        "Z_train": Z_train_ae,
        "Z_test": Z_test_ae,
        "name": f"Autoencoder (MLP {hidden})",
    }

    return results


def run_tsne_grid(
    X_train: np.ndarray,
    _y_train: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Perplexity grid; pick best mean(trustworthiness, continuity) on training embedding."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    best_Z: np.ndarray | None = None
    best_info: dict[str, Any] = {}
    best_score = -np.inf
    rows: list[dict[str, Any]] = []

    for perp in TSNE_PERPLEXITIES:
        if perp >= Xs.shape[0] - 1:
            continue
        t0 = time.perf_counter()
        tsne = TSNE(
            n_components=LATENT_DIM,
            perplexity=perp,
            learning_rate="auto",
            init="pca",
            random_state=RANDOM_STATE,
            max_iter=500 if FAST_MODE else 1000,
            verbose=0,
        )
        Z = tsne.fit_transform(Xs)
        elapsed = time.perf_counter() - t0
        tw = float(trustworthiness(Xs, Z, n_neighbors=NEIGHBORS_QUALITY))
        cont = float(continuity(Xs, Z, n_neighbors=NEIGHBORS_QUALITY))
        qual = 0.5 * (tw + cont)
        stress = kruskal_stress_1(Xs, Z, rng, max_pairs_sample=min(STRESS_SUBSAMPLE, len(Xs)))
        rows.append(
            {
                "perplexity": perp,
                "trustworthiness": tw,
                "continuity": cont,
                "mean(T,C)": qual,
                "kruskal_stress_1": stress,
                "time_s": elapsed,
            }
        )

        if qual > best_score:
            best_score = qual
            best_Z = Z
            best_info = {
                "perplexity": perp,
                "trustworthiness": tw,
                "continuity": cont,
                "kruskal_stress_1": stress,
                "fit_time_s": elapsed,
                "quality_combo": qual,
            }

    print(pd.DataFrame(rows).to_string(index=False))

    assert best_Z is not None
    return best_Z, best_info


def run_umap_grid(
    X_train: np.ndarray,
    _y_train: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not HAS_UMAP:
        raise RuntimeError("Install umap-learn: pip install umap-learn")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    best_Z: np.ndarray | None = None
    best_info: dict[str, Any] = {}
    best_score = -np.inf
    u_rows: list[dict[str, Any]] = []

    for n_neighbors, min_dist in UMAP_GRID:
        if n_neighbors >= Xs.shape[0] - 1:
            continue
        t0 = time.perf_counter()
        reducer = umap.UMAP(
            n_components=LATENT_DIM,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="euclidean",
            random_state=RANDOM_STATE,
            verbose=False,
        )
        Z = reducer.fit_transform(Xs)
        elapsed = time.perf_counter() - t0
        tw = float(trustworthiness(Xs, Z, n_neighbors=NEIGHBORS_QUALITY))
        cont = float(continuity(Xs, Z, n_neighbors=NEIGHBORS_QUALITY))
        qual = 0.5 * (tw + cont)
        stress = kruskal_stress_1(Xs, Z, rng, max_pairs_sample=min(STRESS_SUBSAMPLE, len(Xs)))
        u_rows.append(
            {
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "trustworthiness": tw,
                "continuity": cont,
                "mean(T,C)": qual,
                "kruskal_stress_1": stress,
                "time_s": elapsed,
            }
        )

        if qual > best_score:
            best_score = qual
            best_Z = Z
            best_info = {
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "trustworthiness": tw,
                "continuity": cont,
                "kruskal_stress_1": stress,
                "fit_time_s": elapsed,
                "quality_combo": qual,
            }

    print(pd.DataFrame(u_rows).to_string(index=False))

    assert best_Z is not None
    return best_Z, best_info


def knn_cv_on_embedding(Z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Single embedding: CV only (DR not refitted)."""
    knn = KNeighborsClassifier(n_neighbors=KNN_K)
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(knn, Z, y, cv=cv, n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def plot_embeddings_2d(
    embeddings: dict[str, tuple[np.ndarray, np.ndarray]],
    y_train: np.ndarray,
    out_dir: Path,
) -> None:
    """Scatter plots coloured by class for each method (training set coordinates)."""
    n_classes = len(np.unique(y_train))
    cmap = matplotlib.colormaps["tab10"].resampled(n_classes)

    for key, (Z, title) in embeddings.items():
        if Z.shape[1] != 2:
            continue
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(
            Z[:, 0],
            Z[:, 1],
            c=y_train,
            cmap=cmap,
            s=8,
            alpha=0.75,
            edgecolors="none",
        )
        cbar = plt.colorbar(sc, ax=ax, ticks=range(n_classes))
        cbar.set_label("class")
        ax.set_title(title)
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        fig.tight_layout()
        fig.savefig(out_dir / f"embed_{key}.png", dpi=150)
        plt.close(fig)


def plot_ae_latent_semantics(
    Z: np.ndarray,
    y: np.ndarray,
    stroke: np.ndarray,
    class_names: list[str] | None,
    out_dir: Path,
) -> None:
    """Latent space + correlation with a simple 'ink' proxy (style / thickness exploration)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    n_classes = len(np.unique(y))
    cmap = matplotlib.colormaps["tab10"].resampled(n_classes)
    ax = axes[0]
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap=cmap, s=10, alpha=0.7, edgecolors="none")
    plt.colorbar(sc, ax=ax, ticks=range(n_classes), label="class")
    ax.set_title("AE latent (2D), coloured by class")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")

    ax2 = axes[1]
    sc2 = ax2.scatter(Z[:, 0], Z[:, 1], c=stroke, cmap="viridis", s=10, alpha=0.7, edgecolors="none")
    plt.colorbar(sc2, ax=ax2, label="mean pixel (thickness proxy)")
    ax2.set_title("Same latent, coloured by ink / brightness proxy")
    ax2.set_xlabel("z1")
    ax2.set_ylabel("z2")

    fig.suptitle(
        "Autoencoder latent structure: class clusters vs. coarse style (brightness/thickness proxy)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "ae_latent_semantics.png", dpi=150)
    plt.close(fig)

    # Numeric hints
    if Z.shape[1] >= 2:
        r_z1_class = np.corrcoef(Z[:, 0], y.astype(float))[0, 1]
        r_z1_stroke = np.corrcoef(Z[:, 0], stroke)[0, 1]
        r_z2_stroke = np.corrcoef(Z[:, 1], stroke)[0, 1]
        print(
            f"\n[AE latent vs semantics] corr(z1, class label index)={r_z1_class:.3f} "
            f"(weak; labels are categorical - use visual clusters); "
            f"corr(z1, ink)={r_z1_stroke:.3f}, corr(z2, ink)={r_z2_stroke:.3f}"
        )


def print_complexity_discussion() -> None:
    text = """
--- Computational complexity (typical scalings; n = samples, d = features, k = output dim) ---

PCA (SVD): O(min(n d^2, n^2 d)) time; stores principal directions - memory O(nd + dk).
Kernel PCA: builds n-by-n kernel - O(n^2 d) time and O(n^2) memory; costly for large n.
t-SNE: roughly O(n^2) in naive form; Barnes-Hut ~ O(n log n) per iteration - heavy for large n.
UMAP: neighbor graph + optimization - often similar order to modern t-SNE in practice; lower than
      exact O(n^2) for high d when using approximate NN.
Autoencoder (MLP): O(epochs * n * (#weights)) time; memory O(n * batch) + weights - scales better
      than Kernel PCA in memory when n is huge if batching, but training is iterative.

When autoencoders help vs PCA / t-SNE:
- PCA is optimal only for linear Gaussian reconstruction; it mixes factors (e.g., rotation of digits).
- t-SNE/UMAP optimize visualization quality, not reconstruction, and do not provide a smooth
  parametric mapping for new points unless extra models are trained.
- A nonlinear autoencoder can learn manifolds with bends and separate style factors (stroke width,
  slant) in curved coordinates - useful when labels follow nonlinear generative structure (e.g.,
  lighting, pose on Fashion-MNIST) and you need both reconstruction and usable latent codes for
  downstream models. Kernel PCA helps if the structure is RKHS-captureable but still struggles at
  scale compared to batched neural training.
"""
    print(text)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    n_train = FAST_N_TRAIN if FAST_MODE else FULL_N_TRAIN
    n_test = FAST_N_TEST if FAST_MODE else FULL_N_TEST

    X, y, target_names = load_mnist_like()
    X_train, y_train = stratified_subsample(X, y, n_train, rng)
    X_test, y_test = stratified_subsample(X, y, n_test, rng)

    class_labels = (
        list(target_names)
        if target_names is not None
        else [str(i) for i in range(len(np.unique(y_train)))]
    )
    print(
        f"Dataset: {'Fashion-MNIST' if USE_FASHION_MNIST else 'MNIST'} | "
        f"train={len(y_train)}, test={len(y_test)}, latent_dim={LATENT_DIM}, FAST_MODE={FAST_MODE}"
    )

    # Linear / kernel / AE
    lin = fit_evaluate_pca_kpca_ae(X_train, y_train, X_test, y_test, LATENT_DIM)

    # t-SNE grid
    print("\n--- t-SNE perplexity grid ---")
    Z_tsne, tsne_best = run_tsne_grid(X_train, y_train, rng)
    knn_tsne_m, knn_tsne_s = knn_cv_on_embedding(Z_tsne, y_train)
    tsne_best["knn_cv_acc_mean"] = knn_tsne_m
    tsne_best["knn_cv_acc_std"] = knn_tsne_s
    print(tsne_best)

    # UMAP grid
    print("\n--- UMAP (n_neighbors, min_dist) grid ---")
    if HAS_UMAP:
        Z_umap, umap_best = run_umap_grid(X_train, y_train, rng)
        knn_um_m, knn_um_s = knn_cv_on_embedding(Z_umap, y_train)
        umap_best["knn_cv_acc_mean"] = knn_um_m
        umap_best["knn_cv_acc_std"] = knn_um_s
        print(umap_best)
    else:
        Z_umap = np.zeros((len(y_train), LATENT_DIM))
        umap_best = {"note": "umap-learn not installed"}

    # Figures
    embeddings: dict[str, tuple[np.ndarray, str]] = {
        "pca": (lin["pca"]["Z_train"], "PCA"),
        "kpca": (lin["kpca"]["Z_train"], "Kernel PCA (RBF)"),
        "ae": (lin["ae"]["Z_train"], "Autoencoder"),
        "tsne": (Z_tsne, f"t-SNE (perp={tsne_best.get('perplexity')})"),
        "umap": (Z_umap, f"UMAP (n_neighbors={umap_best.get('n_neighbors')}, min_dist={umap_best.get('min_dist')})"),
    }
    plot_embeddings_2d(embeddings, y_train, FIG_DIR)

    stroke_train = rough_stroke_weight(X_train)
    plot_ae_latent_semantics(
        lin["ae"]["Z_train"],
        y_train,
        stroke_train,
        class_labels,
        FIG_DIR,
    )

    # Summary table
    rows = [
        {
            "method": "PCA",
            "test_recon_mse": lin["pca"]["test_recon_mse"],
            "knn_cv_acc_mean": lin["pca"]["knn_cv_acc_mean"],
            "knn_cv_acc_std": lin["pca"]["knn_cv_acc_std"],
            "trustworthiness": np.nan,
            "continuity": np.nan,
            "kruskal_stress_1": np.nan,
        },
        {
            "method": "Kernel PCA",
            "test_recon_mse": lin["kpca"]["test_recon_mse_pixels"],
            "knn_cv_acc_mean": lin["kpca"]["knn_cv_acc_mean"],
            "knn_cv_acc_std": lin["kpca"]["knn_cv_acc_std"],
            "trustworthiness": np.nan,
            "continuity": np.nan,
            "kruskal_stress_1": np.nan,
        },
        {
            "method": "Autoencoder",
            "test_recon_mse": lin["ae"]["test_recon_mse"],
            "knn_cv_acc_mean": lin["ae"]["knn_cv_acc_mean"],
            "knn_cv_acc_std": lin["ae"]["knn_cv_acc_std"],
            "trustworthiness": np.nan,
            "continuity": np.nan,
            "kruskal_stress_1": np.nan,
        },
        {
            "method": f"t-SNE (perp={tsne_best.get('perplexity')})",
            "test_recon_mse": np.nan,
            "knn_cv_acc_mean": tsne_best.get("knn_cv_acc_mean"),
            "knn_cv_acc_std": tsne_best.get("knn_cv_acc_std"),
            "trustworthiness": tsne_best.get("trustworthiness"),
            "continuity": tsne_best.get("continuity"),
            "kruskal_stress_1": tsne_best.get("kruskal_stress_1"),
        },
    ]
    if HAS_UMAP:
        rows.append(
            {
                "method": f"UMAP (nn={umap_best.get('n_neighbors')}, md={umap_best.get('min_dist')})",
                "test_recon_mse": np.nan,
                "knn_cv_acc_mean": umap_best.get("knn_cv_acc_mean"),
                "knn_cv_acc_std": umap_best.get("knn_cv_acc_std"),
                "trustworthiness": umap_best.get("trustworthiness"),
                "continuity": umap_best.get("continuity"),
                "kruskal_stress_1": umap_best.get("kruskal_stress_1"),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved metrics: {RESULTS_CSV}")
    print(df.to_string(index=False))

    print_complexity_discussion()


if __name__ == "__main__":
    main()
