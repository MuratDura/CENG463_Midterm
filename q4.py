"""
Clustering analysis (K-Means, GMM, DBSCAN, Agglomerative),
including model selection, ensemble clustering, stability, and visualization.

Datasets:
  - Wholesale customers (OpenML id 292, no natural labels): primary internal metrics + viz.
  - sklearn.datasets.load_digits as optdigits-like validation: ARI / NMI / Fowlkes-Mallows.

Outputs: q4_figures/*.png, q4_metrics_summary.csv
Requires: scikit-learn, numpy, pandas, matplotlib, scipy; optional umap-learn for UMAP plots.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
)
from sklearn.datasets import fetch_openml, load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Config ---
RANDOM_STATE = 42
RNG = np.random.default_rng(RANDOM_STATE)
FIG_DIR = Path(__file__).resolve().parent / "q4_figures"
RESULTS_CSV = Path(__file__).resolve().parent / "q4_metrics_summary.csv"

K_RANGE = range(2, 11)
GAP_N_REFS = 20
STABILITY_N_BOOT = 40
STABILITY_FRAC = 0.8
ENSEMBLE_KMEANS_KS = (3, 4, 5, 6)

FAST_MODE = True
FAST_DIGITS_MAX = 1797  # all digits; reduce for quick debug


def _ensure_fig_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_wholesale_scaled() -> tuple[np.ndarray, None]:
    """Wholesale customers; returns X_scaled, y_true (None)."""
    data = fetch_openml(data_id=292, as_frame=False, parser="auto")
    Xraw = data.data
    if hasattr(Xraw, "toarray"):
        X = np.asarray(Xraw.toarray(), dtype=float)
    else:
        X = np.asarray(Xraw, dtype=float)
    # Drop Channel/Region if present (first two columns in classic UCI wholesale layout).
    if X.shape[1] > 6:
        X = X[:, 2:]
    X = StandardScaler().fit_transform(X)
    return X, None


def load_digits_scaled(max_samples: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    d = load_digits()
    X = StandardScaler().fit_transform(d.data)
    y = d.target.astype(int)
    if max_samples is not None and X.shape[0] > max_samples:
        X, y = X[:max_samples], y[:max_samples]
    return X, y


def elbow_inertias(X: np.ndarray, k_range: range, random_state: int) -> tuple[list[int], list[float]]:
    ks, inertias = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X)
        ks.append(k)
        inertias.append(float(km.inertia_))
    return ks, inertias


def gap_statistic(
    X: np.ndarray,
    k_range: range,
    n_refs: int = 20,
    random_state: int = RANDOM_STATE,
) -> tuple[list[float], list[float]]:
    """Tibshirani-style gap statistic (log W_k vs uniform references in feature-wise box)."""
    rng = np.random.default_rng(random_state)
    gaps, sk = [], []
    xmin, xmax = X.min(axis=0), X.max(axis=0)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X)
        log_w = np.log(km.inertia_ + 1e-12)
        ref_logs = []
        for _ in range(n_refs):
            Xr = rng.uniform(xmin, xmax, size=X.shape)
            kmr = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            kmr.fit(Xr)
            ref_logs.append(np.log(kmr.inertia_ + 1e-12))
        ref_logs = np.asarray(ref_logs)
        gap = float(np.mean(ref_logs) - log_w)
        gaps.append(gap)
        sk.append(float(np.std(ref_logs) * np.sqrt(1 + 1.0 / n_refs)))
    return gaps, sk


def pick_k_gap(gaps: list[float], sk: list[float], k_range: list[int]) -> int:
    """Choose smallest k such that gap(k) >= gap(k+1) - s_{k+1} (1-based index in list)."""
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i + 1] - sk[i + 1]:
            return k_range[i]
    return k_range[-1]


def silhouette_per_k(X: np.ndarray, k_range: range, random_state: int) -> dict[int, float]:
    out: dict[int, float] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        lab = km.fit_predict(X)
        if len(np.unique(lab)) < 2:
            out[k] = float("nan")
        else:
            out[k] = float(silhouette_score(X, lab))
    return out


def select_kmeans_k(
    X: np.ndarray,
    k_range: range,
    random_state: int,
) -> tuple[int, dict[str, Any]]:
    ks, inertias = elbow_inertias(X, k_range, random_state)
    gaps, sk = gap_statistic(X, k_range, n_refs=GAP_N_REFS, random_state=random_state)
    k_gap = pick_k_gap(gaps, sk, list(k_range))
    sil = silhouette_per_k(X, k_range, random_state)
    k_sil = max(sil, key=lambda kk: sil[kk] if not np.isnan(sil[kk]) else -np.inf)

    # Simple elbow: max distance from line (first,last) in (k, inertia) space
    k_arr = np.asarray(ks, dtype=float)
    w_arr = np.asarray(inertias, dtype=float)
    line = w_arr[0] + (w_arr[-1] - w_arr[0]) * (k_arr - k_arr[0]) / (k_arr[-1] - k_arr[0] + 1e-12)
    dist = np.abs(w_arr - line)
    k_elbow = int(ks[int(np.argmax(dist))])

    meta = {
        "ks": ks,
        "inertias": inertias,
        "gaps": gaps,
        "gap_sk": sk,
        "silhouette": sil,
        "k_gap": k_gap,
        "k_silhouette": k_sil,
        "k_elbow_heuristic": k_elbow,
    }
    # Prefer gap rule; fallback silhouette
    k_final = k_gap if k_gap in k_range else k_sil
    return k_final, meta


def fit_gmm_select_components(
    X: np.ndarray,
    n_range: range,
    random_state: int,
) -> tuple[int, GaussianMixture, pd.DataFrame]:
    rows = []
    best_bic = np.inf
    best_n = n_range.start
    best_gmm: GaussianMixture | None = None
    for n in n_range:
        gmm = GaussianMixture(
            n_components=n,
            covariance_type="full",
            random_state=random_state,
            n_init=3,
            max_iter=200,
        )
        gmm.fit(X)
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        rows.append({"n_components": n, "bic": bic, "aic": aic})
        if bic < best_bic:
            best_bic = bic
            best_n = n
            best_gmm = gmm
    assert best_gmm is not None
    return best_n, best_gmm, pd.DataFrame(rows)


def k_distance_graph(X: np.ndarray, min_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sorted distances to min_samples-th neighbor (excluding self)."""
    nn = NearestNeighbors(n_neighbors=min_samples + 1, metric="euclidean")
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    kth = dists[:, -1]
    kth_sorted = np.sort(kth)
    return np.arange(len(kth_sorted)), kth_sorted


def suggest_eps_from_kdist(kth_sorted: np.ndarray, percentile: float = 90.0) -> float:
    """Heuristic eps at high curvature region: use percentile of k-distances."""
    return float(np.percentile(kth_sorted, percentile))


def fit_dbscan_tuned(
    X: np.ndarray,
    min_samples: int,
    eps_candidates: np.ndarray | None = None,
) -> tuple[DBSCAN, float, dict[str, Any]]:
    idx, kth_sorted = k_distance_graph(X, min_samples)
    if eps_candidates is None:
        eps_candidates = np.linspace(
            float(np.percentile(kth_sorted, 70)),
            float(np.percentile(kth_sorted, 98)),
            25,
        )
    best_eps = eps_candidates[0]
    best_score = -np.inf
    best_db: DBSCAN | None = None
    for eps in eps_candidates:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        lab = db.fit_predict(X)
        nlab = len(set(lab)) - (1 if -1 in lab else 0)
        if nlab < 2:
            continue
        mask = lab >= 0
        if np.sum(mask) < 2:
            continue
        try:
            sil = silhouette_score(X[mask], lab[mask])
        except Exception:
            continue
        if sil > best_score:
            best_score = sil
            best_eps = float(eps)
            best_db = db
    if best_db is None:
        best_eps = suggest_eps_from_kdist(kth_sorted, 85)
        best_db = DBSCAN(eps=best_eps, min_samples=min_samples)
        best_db.fit(X)
    meta = {"kdist_index": idx, "kdist_sorted": kth_sorted, "eps_chosen": best_eps}
    return best_db, best_eps, meta


def internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    ul = np.unique(labels)
    if len(ul) < 2 or (labels == labels[0]).all():
        return {
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
        }
    mask = labels >= 0
    if np.sum(mask) < 2:
        return {
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
        }
    Xe, ye = X[mask], labels[mask]
    return {
        "silhouette": float(silhouette_score(Xe, ye)),
        "calinski_harabasz": float(calinski_harabasz_score(Xe, ye)),
        "davies_bouldin": float(davies_bouldin_score(Xe, ye)),
    }


def external_metrics(y_true: np.ndarray, labels: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    if mask is not None:
        y_true = y_true[mask]
        labels = labels[mask]
    return {
        "ari": float(adjusted_rand_score(y_true, labels)),
        "nmi": float(normalized_mutual_info_score(y_true, labels)),
        "fmi": float(fowlkes_mallows_score(y_true, labels)),
    }


def clustering_stability_ari(
    X: np.ndarray,
    n_clusters: int,
    n_boot: int,
    frac: float,
    random_state: int,
) -> tuple[float, float]:
    """
    Bootstrap frac of samples per run; KMeans with fixed k.
    For each pair of runs, restrict to intersection of sampled indices and compute ARI.
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    size = int(round(frac * n))
    label_sets: list[tuple[np.ndarray, np.ndarray]] = []
    for b in range(n_boot):
        idx = rng.choice(n, size=size, replace=False)
        km = KMeans(n_clusters=n_clusters, random_state=random_state + b, n_init=10)
        sub_lab = km.fit_predict(X[idx])
        label_sets.append((idx, sub_lab))

    aris: list[float] = []
    for i in range(n_boot):
        for j in range(i + 1, n_boot):
            idx_i, li = label_sets[i]
            idx_j, lj = label_sets[j]
            inter = np.intersect1d(idx_i, idx_j, assume_unique=False)
            if inter.size < n_clusters * 2:
                continue
            map_i = {v: k for k, v in enumerate(idx_i)}
            map_j = {v: k for k, v in enumerate(idx_j)}
            li_a = np.array([li[map_i[v]] for v in inter])
            lj_a = np.array([lj[map_j[v]] for v in inter])
            aris.append(adjusted_rand_score(li_a, lj_a))

    if not aris:
        return float("nan"), float("nan")
    return float(np.mean(aris)), float(np.std(aris))


def coassociation_ensemble(
    X: np.ndarray,
    base_labels: list[np.ndarray],
    n_clusters_out: int,
) -> np.ndarray:
    """Average co-association then average-linkage agglomerative on distance = 1 - S."""
    n = X.shape[0]
    # Co-association matrix: entry (i,j) stores fraction of base runs where i and j co-cluster.
    acc = np.zeros((n, n), dtype=float)
    for lab in base_labels:
        for c in np.unique(lab):
            if c < 0:
                continue
            idx = np.where(lab == c)[0]
            if idx.size:
                acc[np.ix_(idx, idx)] += 1.0
    acc /= max(len(base_labels), 1)
    dist = 1.0 - acc
    np.fill_diagonal(dist, 0.0)
    clust = AgglomerativeClustering(
        n_clusters=n_clusters_out,
        metric="precomputed",
        linkage="average",
    )
    return clust.fit_predict(dist)


def run_pipeline(
    name: str,
    X: np.ndarray,
    y_true: np.ndarray | None,
    random_state: int,
) -> list[dict[str, Any]]:
    _ensure_fig_dir()
    rows: list[dict[str, Any]] = []

    # --- K-Means selection ---
    k_km, km_meta = select_kmeans_k(X, K_RANGE, random_state)
    km = KMeans(n_clusters=k_km, random_state=random_state, n_init=10)
    labels_km = km.fit_predict(X)
    m = internal_metrics(X, labels_km)
    row_km = {"dataset": name, "method": f"KMeans_k={k_km}", **m}
    if y_true is not None:
        row_km.update(**{f"ext_{k}": v for k, v in external_metrics(y_true, labels_km).items()})
    rows.append(row_km)

    # Plots: elbow + gap + silhouette
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(km_meta["ks"], km_meta["inertias"], "o-")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia (WCSS)")
    axes[0].set_title("Elbow (K-Means)")
    axes[1].plot(list(K_RANGE), km_meta["gaps"], "o-", label="Gap")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Gap(k)")
    axes[1].set_title("Gap statistic")
    sil_vals = [km_meta["silhouette"][k] for k in K_RANGE]
    axes[2].plot(list(K_RANGE), sil_vals, "o-")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("Silhouette")
    axes[2].set_title("Silhouette vs k")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}_kmeans_selection.png", dpi=150)
    plt.close(fig)

    # --- GMM ---
    n_gmm, gmm, gmm_df = fit_gmm_select_components(X, K_RANGE, random_state)
    labels_gmm = gmm.predict(X)
    row_gmm = {"dataset": name, "method": f"GMM_n={n_gmm}", **internal_metrics(X, labels_gmm)}
    if y_true is not None:
        row_gmm.update(**{f"ext_{k}": v for k, v in external_metrics(y_true, labels_gmm).items()})
    rows.append(row_gmm)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(gmm_df["n_components"], gmm_df["bic"], "o-", label="BIC")
    ax.plot(gmm_df["n_components"], gmm_df["aic"], "s--", label="AIC")
    ax.set_xlabel("n_components")
    ax.set_ylabel("Information criterion")
    ax.legend()
    ax.set_title("GMM model selection")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}_gmm_bic_aic.png", dpi=150)
    plt.close(fig)

    # --- DBSCAN ---
    min_samples = max(3, int(round(np.log(X.shape[0] + 1))))
    db, eps_used, db_meta = fit_dbscan_tuned(X, min_samples=min_samples)
    labels_db = db.labels_
    row_db = {"dataset": name, "method": f"DBSCAN_eps={eps_used:.4f}", **internal_metrics(X, labels_db)}
    if y_true is not None:
        row_db.update(**{f"ext_{k}": v for k, v in external_metrics(y_true, labels_db).items()})
    rows.append(row_db)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(db_meta["kdist_index"], db_meta["kdist_sorted"], lw=1)
    ax.axhline(eps_used, color="C1", ls="--", label=f"eps={eps_used:.4f}")
    ax.set_xlabel("Points (sorted)")
    ax.set_ylabel(f"{min_samples}-distance")
    ax.set_title("k-distance graph (DBSCAN)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}_dbscan_kdistance.png", dpi=150)
    plt.close(fig)

    # --- Agglomerative Ward + dendrogram (sample if large) ---
    n_link = min(200, X.shape[0])
    rng = np.random.default_rng(random_state)
    sub_idx = rng.choice(X.shape[0], size=n_link, replace=False)
    Xs = X[sub_idx]
    Z = linkage(Xs, method="ward")
    max_d = Z[-1, 2] * 0.7
    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(Z, ax=ax, color_threshold=max_d, no_labels=True)
    ax.set_title(f"Agglomerative (Ward) dendrogram (n={n_link} subsample)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}_dendrogram_ward.png", dpi=150)
    plt.close(fig)

    n_agg = int(np.median([k_km, n_gmm, max(2, len(np.unique(labels_db[labels_db >= 0])))]))
    n_agg = int(np.clip(n_agg, 2, 15))
    agg = AgglomerativeClustering(n_clusters=n_agg, linkage="ward")
    labels_agg = agg.fit_predict(X)
    row_agg = {"dataset": name, "method": f"Agglomerative_Ward_k={n_agg}", **internal_metrics(X, labels_agg)}
    if y_true is not None:
        row_agg.update(**{f"ext_{k}": v for k, v in external_metrics(y_true, labels_agg).items()})
    rows.append(row_agg)

    # --- Stability ---
    mean_ari, std_ari = clustering_stability_ari(
        X, n_clusters=k_km, n_boot=STABILITY_N_BOOT if not FAST_MODE else 15, frac=STABILITY_FRAC, random_state=random_state
    )
    stab_row = {
        "dataset": name,
        "method": "Stability_KMeans_bootstrap_ARI",
        "silhouette": mean_ari,
        "calinski_harabasz": std_ari,
        "davies_bouldin": float("nan"),
    }
    rows.append(stab_row)
    print(
        f"[{name}] K-Means stability (mean ARI between bootstrap clusterings, intersected points): "
        f"{mean_ari:.4f} +/- {std_ari:.4f}"
    )

    # --- Ensemble: co-association from multiple KMeans + GMM + DBSCAN ---
    base_labs: list[np.ndarray] = []
    for kk in ENSEMBLE_KMEANS_KS:
        km_e = KMeans(n_clusters=kk, random_state=random_state, n_init=10)
        base_labs.append(km_e.fit_predict(X))
    base_labs.append(labels_gmm.copy())
    base_labs.append(labels_db.copy())

    n_ens = int(np.round(np.median([k_km, n_gmm, n_agg])))
    n_ens = int(np.clip(n_ens, 2, 12))
    try:
        labels_ens = coassociation_ensemble(X, base_labs, n_clusters_out=n_ens)
    except Exception:
        labels_ens = labels_km.copy()

    row_ens = {"dataset": name, "method": f"Ensemble_CSPA_k={n_ens}", **internal_metrics(X, labels_ens)}
    if y_true is not None:
        row_ens.update(**{f"ext_{k}": v for k, v in external_metrics(y_true, labels_ens).items()})
    rows.append(row_ens)

    # --- 2D visualization: PCA + optional UMAP ---
    pca2 = PCA(n_components=2, random_state=random_state)
    Xp = pca2.fit_transform(X)

    def _scatter_2d(xy: np.ndarray, labels: np.ndarray, title: str, fname: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        ulab = np.unique(labels)
        for c in ulab:
            m = labels == c
            ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.7, label=str(c))
        ax.set_title(title)
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        if len(ulab) <= 12:
            ax.legend(markerscale=2, fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(FIG_DIR / fname, dpi=150)
        plt.close(fig)

    _scatter_2d(Xp, labels_km, "K-Means (PCA-2D)", f"{name}_pca_kmeans.png")
    _scatter_2d(Xp, labels_gmm, "GMM (PCA-2D)", f"{name}_pca_gmm.png")
    _scatter_2d(Xp, labels_db, "DBSCAN (PCA-2D)", f"{name}_pca_dbscan.png")
    _scatter_2d(Xp, labels_agg, "Agglomerative Ward (PCA-2D)", f"{name}_pca_agg.png")
    _scatter_2d(Xp, labels_ens, "Ensemble CSPA (PCA-2D)", f"{name}_pca_ensemble.png")

    if HAS_UMAP and (FAST_MODE is False or X.shape[0] < 2500):
        redu = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1)
        Xu = redu.fit_transform(X)
        _scatter_2d(Xu, labels_km, "K-Means (UMAP)", f"{name}_umap_kmeans.png")
        _scatter_2d(Xu, labels_db, "DBSCAN (UMAP)", f"{name}_umap_dbscan.png")
    elif not HAS_UMAP:
        tsne = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto")
        Xt = tsne.fit_transform(X)
        _scatter_2d(Xt, labels_km, "K-Means (t-SNE fallback)", f"{name}_tsne_kmeans.png")

    return rows


def print_assumptions_discussion() -> None:
    text = """
=== Method assumptions and effect of violations ===
- K-Means: assumes isotropic (spherical) clusters of similar size; minimizes within-cluster
  variance. Elongated / uneven clusters get split or merged; outliers skew centroids.
- GMM: assumes each cluster is a Gaussian (full covariance can model ellipsoids). Too few
  components underfit; many components can over-segment. Heavy tails and non-Gaussian shapes
  reduce fit quality.
- DBSCAN: density-connected regions of similar density; can find arbitrary shapes. Fails if
  clusters have very different densities or if global eps cannot separate gaps; noise label -1.
- Agglomerative (Ward): tends to merge clusters to minimize variance increase; often prefers
  compact, similarly-sized merge steps; sensitive to noise and feature scaling.
- Ensemble (co-association / CSPA): combines soft agreement across runs; can smooth single-
  method failures but may blur fine structure if base partitions disagree strongly.
"""
    print(text)


def main() -> None:
    _ensure_fig_dir()
    all_rows: list[dict[str, Any]] = []

    X_w, _ = load_wholesale_scaled()
    all_rows.extend(run_pipeline("wholesale", X_w, y_true=None, random_state=RANDOM_STATE))

    nmax = FAST_DIGITS_MAX if FAST_MODE else 1797
    X_d, y_d = load_digits_scaled(max_samples=nmax)
    all_rows.extend(run_pipeline("digits_optdigits_like", X_d, y_true=y_d, random_state=RANDOM_STATE))

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved metrics: {RESULTS_CSV}")

    ext_df = df[df["dataset"] == "digits_optdigits_like"].copy()
    ext_df = ext_df[ext_df["ext_ari"].notna()]
    if not ext_df.empty:
        cmp_cols = ["method", "ext_ari", "ext_nmi", "ext_fmi"]
        print("\nDigits (labeled): external validation (Ensemble vs individual):")
        print(ext_df[cmp_cols].to_string(index=False))

    print_assumptions_discussion()


if __name__ == "__main__":
    main()
