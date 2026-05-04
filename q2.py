"""
Binary classification under extreme class imbalance with cost-sensitive learning and calibration.

Dataset: ULB credit card fraud via OpenML (data_id=1597); minority rate ~0.17% (IR ~578).

Contents:
  - Four classifiers: LogisticRegression, RandomForest, XGBoost (scale_pos_weight per fold),
    MLP (balanced sample_weight per fold).
  - Metrics from StratifiedKFold CV (pipelines prevent leakage): precision/recall/F1 macro/micro,
    ROC-AUC, PR-AUC, MCC, balanced accuracy.
  - Resampling inside Pipeline: none, SMOTE, ADASYN, RandomUnderSampler; compared to a run
    without class weights / scale_pos_weight.
  - CalibratedClassifierCV (Platt=sigmoid, isotonic) on the two best model types by PR-AUC;
    Brier score + reliability diagrams in ./q2_figures/.
  - PR curve, F1-optimal and cost-weighted thresholds, confusion matrices, narrative on FN vs FP.

Requires: pip install imbalanced-learn  (imblearn)

Use FAST_MODE=False for full data and stronger conclusions (longer runtime).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / avoid Tk threading issues on Windows
from typing import Any, Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    make_scorer,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
FIG_DIR = Path(__file__).resolve().parent / "q2_figures"
RESULTS_DIR = Path(__file__).resolve().parent

# Set FAST_MODE=True for quick runs (stratified subsample).
FAST_MODE = True
FAST_N_SAMPLES = 60_000
CV_SPLITS = 3 if FAST_MODE else 5

np.random.seed(RANDOM_STATE)


def matthews_corrcoef_scorer(estimator, X, y) -> float:
    y_pred = estimator.predict(X)
    return matthews_corrcoef(y, y_pred)


def load_credit_card_fraud() -> tuple[np.ndarray, np.ndarray]:
    """ULB credit card fraud via OpenML (data_id=1597)."""
    bunch = fetch_openml(data_id=1597, as_frame=True, parser="auto")
    X = bunch.data.to_numpy(dtype=np.float64)
    y = bunch.target.astype(np.int64).to_numpy()
    return X, y


def imbalance_ratio(y: np.ndarray) -> float:
    n0 = int(np.sum(y == 0))
    n1 = int(np.sum(y == 1))
    return n0 / max(n1, 1)


def scale_pos_weight(y: np.ndarray) -> float:
    n_neg = int(np.sum(y == 0))
    n_pos = max(int(np.sum(y == 1)), 1)
    return n_neg / n_pos


class XGBCostSensitiveClassifier(ClassifierMixin, BaseEstimator):
    """XGBoost with scale_pos_weight recomputed from each fit() call (correct under CV)."""

    def __init__(
        self,
        *,
        use_cost_sensitive: bool = True,
        n_estimators: int = 120,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        eval_metric: str = "logloss",
        random_state: int = RANDOM_STATE,
        tree_method: str = "hist",
    ):
        self.use_cost_sensitive = use_cost_sensitive
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.tree_method = tree_method

    def fit(self, X: np.ndarray, y: np.ndarray):
        spw = scale_pos_weight(y) if self.use_cost_sensitive else 1.0
        self.estimator_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            eval_metric=self.eval_metric,
            random_state=self.random_state,
            tree_method=self.tree_method,
            scale_pos_weight=spw,
        )
        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict_proba(X)


class MLPCostSensitiveClassifier(ClassifierMixin, BaseEstimator):
    """MLP with balanced sample_weight at fit (class-weight behaviour for neural nets)."""

    def __init__(
        self,
        *,
        use_cost_sensitive: bool = True,
        hidden_layer_sizes: tuple[int, ...] = (128, 64),
        max_iter: int = 400,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 15,
        random_state: int = RANDOM_STATE,
    ):
        self.use_cost_sensitive = use_cost_sensitive
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.estimator_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state,
        )
        sw = (
            compute_sample_weight("balanced", y)
            if self.use_cost_sensitive
            else None
        )
        self.estimator_.fit(X, y, sample_weight=sw)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict_proba(X)


def make_base_estimators(
    _y_train: np.ndarray, *, use_cost_sensitive: bool
) -> dict[str, Any]:
    cw_lr = "balanced" if use_cost_sensitive else None
    cw_rf = "balanced" if use_cost_sensitive else None
    est: dict[str, Any] = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight=cw_lr,
            solver="lbfgs",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200 if not FAST_MODE else 100,
            max_depth=12,
            class_weight=cw_rf,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBCostSensitiveClassifier(
            use_cost_sensitive=use_cost_sensitive,
            n_estimators=(200 if not FAST_MODE else 120),
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            tree_method="hist",
        ),
        "MLP": MLPCostSensitiveClassifier(
            use_cost_sensitive=use_cost_sensitive,
            hidden_layer_sizes=(128, 64),
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=RANDOM_STATE,
        ),
    }
    return est


SamplerKind = Literal["none", "smote", "adasyn", "rus"]


def make_sampler(kind: SamplerKind, *, random_state: int = RANDOM_STATE):
    if kind == "none":
        return None
    if kind == "smote":
        return SMOTE(random_state=random_state, k_neighbors=5)
    if kind == "adasyn":
        return ADASYN(random_state=random_state, n_neighbors=5)
    if kind == "rus":
        return RandomUnderSampler(random_state=random_state)
    raise ValueError(kind)


def make_pipeline(
    sampler: Any | None,
    clf: Any,
) -> ImbPipeline:
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("clf", clf))
    return ImbPipeline(steps)


def build_scoring_dict() -> dict[str, Callable]:
    return {
        "average_precision": "average_precision",
        "roc_auc": "roc_auc",
        "f1_macro": "f1_macro",
        "f1_micro": "f1_micro",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "balanced_accuracy": "balanced_accuracy",
        "mcc": make_scorer(matthews_corrcoef_scorer),
    }


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    *,
    use_cost_sensitive: bool,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = build_scoring_dict()
    rows: list[dict[str, Any]] = []

    # Evaluate each sampler/model pair under the same stratified folds for fair comparison.
    for sname in ("none", "smote", "adasyn", "rus"):
        sampler = make_sampler(sname)  # type: ignore[arg-type]
        est_map = make_base_estimators(y, use_cost_sensitive=use_cost_sensitive)
        for model_name, base in est_map.items():
            pipe = make_pipeline(sampler, clone(base))
            try:
                out = cross_validate(
                    pipe,
                    X,
                    y,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=1,
                    return_train_score=False,
                )
            except ValueError as e:
                # ADASYN can fail on tiny minority pockets in a fold
                rows.append(
                    {
                        "sampler": sname,
                        "model": model_name,
                        "cost_sensitive": use_cost_sensitive,
                        "error": str(e),
                    }
                )
                continue

            for metric in scoring:
                key = f"test_{metric}"
                mean_v = float(np.mean(out[key]))
                std_v = float(np.std(out[key]))
                rows.append(
                    {
                        "sampler": sname,
                        "model": model_name,
                        "cost_sensitive": use_cost_sensitive,
                        "metric": metric,
                        "mean": mean_v,
                        "std": std_v,
                    }
                )
    return pd.DataFrame(rows)


def pivot_cv_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "metric" not in df.columns:
        return df
    ok = df[df["metric"].notna() & df["mean"].notna()].copy()
    if ok.empty:
        return df
    wide = ok.pivot_table(
        index=["sampler", "model"],
        columns="metric",
        values="mean",
        aggfunc="first",
    )
    out = wide.reset_index()
    return out


def pick_top_two_pipelines(
    wide: pd.DataFrame,
) -> list[tuple[str, str]]:
    """Return [(sampler, model), ...] for top-2 by average_precision (PR-AUC)."""
    if "average_precision" not in wide.columns:
        return [("none", "LogisticRegression"), ("smote", "XGBoost")]
    d = wide.sort_values("average_precision", ascending=False).head(4)
    seen: set[str] = set()
    picked: list[tuple[str, str]] = []
    for _, r in d.iterrows():
        key = str(r["model"])
        if key in seen:
            continue
        seen.add(key)
        picked.append((str(r["sampler"]), key))
        if len(picked) >= 2:
            break
    if len(picked) < 2:
        picked.append(("none", "RandomForest"))
    return picked[:2]


def fit_calibrated(
    base_pipeline: ImbPipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: Literal["sigmoid", "isotonic"],
) -> CalibratedClassifierCV:
    cal = CalibratedClassifierCV(
        base_pipeline,
        method=method,
        cv=3,
        n_jobs=1,
    )
    cal.fit(X_train, y_train)
    return cal


def reliability_plot(
    y_true: np.ndarray,
    prob_before: np.ndarray,
    prob_after: np.ndarray,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    prob_true_b, prob_pred_b = calibration_curve(y_true, prob_before, n_bins=10, strategy="quantile")
    prob_true_a, prob_pred_a = calibration_curve(y_true, prob_after, n_bins=10, strategy="quantile")
    ax.plot([0, 1], [0, 1], "k:", label="Perfect")
    ax.plot(prob_pred_b, prob_true_b, "s-", label="Before calibration")
    ax.plot(prob_pred_a, prob_true_a, "o-", label="After calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pr_curve_with_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str,
    path: Path,
    *,
    cost_fn: float,
    cost_fp: float,
) -> tuple[float, float]:
    """Return (f1_best_threshold, cost_best_threshold)."""
    precision, recall, thr = precision_recall_curve(y_true, y_prob)
    # F1 at each threshold from PR curve construction
    f1s = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_i = int(np.nanargmax(f1s))
    t_f1 = float(thr[best_i]) if best_i < len(thr) else 0.5

    costs = []
    ts = np.linspace(1e-6, 1 - 1e-6, 400)
    for t in ts:
        y_hat = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        costs.append(fn * cost_fn + fp * cost_fp)
    t_cost = float(ts[int(np.argmin(costs))])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label="PR curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return t_f1, t_cost


def confusion_plots(
    y_true: np.ndarray,
    y_pred_default: np.ndarray,
    y_pred_t_f1: np.ndarray,
    labels: tuple[str, str],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, yp, name in zip(
        axes,
        (y_pred_default, y_pred_t_f1),
        ("Default 0.5 threshold", "F1-max threshold"),
    ):
        cm = confusion_matrix(y_true, yp, labels=labels)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(int(v)), ha="center", va="center", color="black")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    X_raw, y_raw = load_credit_card_fraud()
    ir = imbalance_ratio(y_raw)
    n_pos = int(np.sum(y_raw == 1))
    n_neg = int(np.sum(y_raw == 0))
    print("=== Dataset: Credit Card Fraud (OpenML 1597 / ULB) ===")
    print(f"Samples: {len(y_raw)}, features: {X_raw.shape[1]}")
    print(f"Class 0 (legitimate): {n_neg}, Class 1 (fraud): {n_pos}")
    print(f"Minority rate: {100 * n_pos / len(y_raw):.4f}%")
    print(f"Imbalance ratio (majority/minority): IR = {ir:.2f}")

    if FAST_MODE and len(y_raw) > FAST_N_SAMPLES:
        idx = np.arange(len(y_raw))
        rng = np.random.RandomState(RANDOM_STATE)
        # stratified subsample
        i0 = rng.choice(idx[y_raw == 0], size=min(FAST_N_SAMPLES // 2, n_neg), replace=False)
        i1 = rng.choice(idx[y_raw == 1], size=min(FAST_N_SAMPLES // 2, n_pos), replace=False)
        sub = np.sort(np.concatenate([i0, i1]))
        X_raw, y_raw = X_raw[sub], y_raw[sub]
        print(
            f"\n[FAST_MODE] Stratified subsample to n={len(y_raw)} for quicker CV "
            f"(set FAST_MODE=False for full data)."
        )
        print(f"New IR ~ {imbalance_ratio(y_raw):.2f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y_raw,
        test_size=0.25,
        stratify=y_raw,
        random_state=RANDOM_STATE,
    )

    print("\n--- Cross-validation: cost-sensitive + resampling pipelines ---")
    df_cs = run_cross_validation(X_train, y_train, use_cost_sensitive=True)
    df_cs.to_csv(RESULTS_DIR / "q2_cv_long_cost_sensitive.csv", index=False)
    wide_cs = pivot_cv_table(df_cs)
    if not wide_cs.empty:
        wide_cs.to_csv(RESULTS_DIR / "q2_cv_summary_cost_sensitive.csv", index=False)
        print(wide_cs.to_string(index=False))

    print("\n--- Cross-validation: NO class weights / NO scale_pos_weight (resampling only) ---")
    df_plain = run_cross_validation(X_train, y_train, use_cost_sensitive=False)
    df_plain.to_csv(RESULTS_DIR / "q2_cv_long_no_cost_sensitive.csv", index=False)
    wide_plain = pivot_cv_table(df_plain)
    if not wide_plain.empty:
        wide_plain.to_csv(RESULTS_DIR / "q2_cv_summary_no_cost_sensitive.csv", index=False)
        print(wide_plain.to_string(index=False))

    # Discussion table: best per sampler
    print("\n--- Best PR-AUC (average_precision) per resampling strategy (cost-sensitive) ---")
    if not wide_cs.empty and "average_precision" in wide_cs.columns:
        best_per = wide_cs.loc[
            wide_cs.groupby("sampler")["average_precision"].idxmax()
        ].reset_index(drop=True)
        show_cols = [
            "sampler",
            "model",
            "average_precision",
            "roc_auc",
            "f1_macro",
        ]
        if "mcc" in best_per.columns:
            show_cols.append("mcc")
        print(best_per[show_cols])

    top_two = pick_top_two_pipelines(wide_cs if not wide_cs.empty else wide_plain)
    print("\n--- Calibration on hold-out test (top-2 model families from CV) ---")
    cal_rows: list[dict[str, Any]] = []
    for method in ("sigmoid", "isotonic"):
        for (sname, mname) in top_two:
            sampler = make_sampler(sname)  # type: ignore[arg-type]
            est_map = make_base_estimators(y_train, use_cost_sensitive=True)
            base = clone(est_map[mname])
            pipe = make_pipeline(sampler, base)
            pipe.fit(X_train, y_train)
            prob_uncal = pipe.predict_proba(X_test)[:, 1]

            cal = fit_calibrated(
                make_pipeline(sampler, clone(est_map[mname])),
                X_train,
                y_train,
                method=method,  # type: ignore[arg-type]
            )
            prob_cal = cal.predict_proba(X_test)[:, 1]

            brier_u = brier_score_loss(y_test, prob_uncal)
            brier_c = brier_score_loss(y_test, prob_cal)
            cal_rows.append(
                {
                    "sampler": sname,
                    "model": mname,
                    "method": method,
                    "brier_uncalibrated": brier_u,
                    "brier_calibrated": brier_c,
                }
            )
            tag = f"{mname}_{sname}_{method}"
            reliability_plot(
                y_test,
                prob_uncal,
                prob_cal,
                f"{mname} ({sname}) — {method}",
                FIG_DIR / f"reliability_{tag}.png",
            )
    pd.DataFrame(cal_rows).to_csv(RESULTS_DIR / "q2_calibration_brier.csv", index=False)
    print(pd.DataFrame(cal_rows))

    # PR curve + thresholds on best single pipeline (first of top_two)
    s0, m0 = top_two[0]
    sampler0 = make_sampler(s0)  # type: ignore[arg-type]
    em = make_base_estimators(y_train, use_cost_sensitive=True)
    best_pipe = make_pipeline(sampler0, clone(em[m0]))
    best_pipe.fit(X_train, y_train)
    p_test = best_pipe.predict_proba(X_test)[:, 1]

    cost_fn, cost_fp = 50.0, 1.0  # fraud miss often >> false alarm
    t_f1, t_cost = pr_curve_with_thresholds(
        y_test,
        p_test,
        f"PR curve — {m0} ({s0})",
        FIG_DIR / f"pr_curve_{m0}_{s0}.png",
        cost_fn=cost_fn,
        cost_fp=cost_fp,
    )
    print(f"\nOptimal threshold (F1 on PR support): {t_f1:.4f}")
    print(
        f"Cost-weighted optimal threshold (cost_FN={cost_fn}, cost_FP={cost_fp}): {t_cost:.4f}"
    )

    y_default = (p_test >= 0.5).astype(int)
    y_f1 = (p_test >= t_f1).astype(int)
    confusion_plots(
        y_test,
        y_default,
        y_f1,
        (0, 1),
        FIG_DIR / f"confusion_{m0}_{s0}.png",
    )

    # ROC for reference
    fpr, tpr, _ = roc_curve(y_test, p_test)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC={roc_auc_score(y_test, p_test):.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC — {m0} ({s0})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"roc_{m0}_{s0}.png", dpi=150)
    plt.close(fig)

    print("\n=== Real-world cost discussion (fraud) ===")
    tn, fp, fn, tp = confusion_matrix(y_test, y_default).ravel()
    print(
        "At threshold 0.5: "
        f"TN={tn}, FP={fp}, FN={fn}, TP={tp}. "
        "False negatives are missed fraud (direct monetary loss + liability). "
        "False positives inconvenience legitimate customers (reviews, blocks). "
        "When the cost of a missed fraud far exceeds checking a legitimate transaction, "
        "we prefer higher recall (lower threshold or cost-weighted decision); PR-AUC and "
        "cost-optimal thresholds reflect this better than raw accuracy under imbalance."
    )


if __name__ == "__main__":
    main()
