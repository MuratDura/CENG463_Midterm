"""
Regression pipeline with regularization, feature engineering, and model selection.
California Housing dataset: loading, EDA, model comparison, and hold-out evaluation.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from scipy.stats import loguniform, randint, ttest_rel, uniform, wilcoxon
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import (
    ElasticNetCV,
    HuberRegressor,
    LassoCV,
    LinearRegression,
    Ridge,
    RidgeCV,
)
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    RepeatedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from xgboost import XGBRegressor

RANDOM_STATE = 42
FIG_DIR = Path(__file__).resolve().parent / "eda_figures"
RESULTS_CSV = Path(__file__).resolve().parent / "model_cv_metrics.csv"
RESULTS_FOLDS_CSV = Path(__file__).resolve().parent / "model_cv_folds.csv"
PAIRWISE_TESTS_CSV = Path(__file__).resolve().parent / "model_pairwise_tests.csv"
FINAL_HOLDOUT_METRICS_CSV = Path(__file__).resolve().parent / "model_holdout_test_metrics.csv"
# Single held-out test fraction (same split seed as CV shuffle for reproducibility narrative).
HOLDOUT_TEST_SIZE = 0.2
HUBER_REGRESSOR_NAME = "HuberRegressor (robust)"

# --- Feature engineering (polynomial deg 2+3 via degree=3, log skewed, RFE or RF top-k) ---
USE_FEATURE_ENGINEERING = True
FE_POLY_DEGREE = 3  # includes all degree-1,2,3 monomials (pairwise interactions in 2nd order)
FE_SKEW_THRESHOLD = 0.5  # |skew| above this -> log1p on that column (if min > -1)
FE_TARGET_SKEW_LOG = True  # log1p target when |skew(y)| exceeds FE_TARGET_SKEW_THRESHOLD
FE_TARGET_SKEW_THRESHOLD = 0.25
FE_N_TOP_FEATURES = 50
FE_RFE_STEP = 25
# "rfe" = recursive feature elimination (Ridge); "rf" = top-k by RandomForest importances (faster)
FE_SELECTION_METHOD: str = "rf"

# Training objective / loss used by each estimator (for the report)
MODEL_TRAINING_LOSS: dict[str, str] = {
    "Linear Regression": (
        "Ordinary Least Squares: minimize sum of squared residuals (MSE), no penalty."
    ),
    "Ridge (RidgeCV)": (
        "Linear least squares with L2 penalty: minimize ||y-Xb||^2 + alpha*||b||_2^2; "
        "alpha chosen by inner CV."
    ),
    "Lasso (LassoCV)": (
        "Linear model with L1 penalty: minimize (1/(2n))||y-Xb||^2 + alpha*||b||_1; "
        "alpha chosen by inner CV."
    ),
    "Elastic Net (ElasticNetCV)": (
        "Minimize (1/(2n))||y-Xb||^2 + alpha*(rho*||b||_1 + (1-rho)/2*||b||_2^2); "
        "l1_ratio rho and alpha chosen by inner CV."
    ),
    "XGBoost (tuned)": (
        "Gradient boosting with objective reg:squarederror (squared loss on residuals); "
        "hyperparameters via RandomizedSearchCV on each outer-train fold only."
    ),
    HUBER_REGRESSOR_NAME: (
        "Robust regression: Huber loss (quadratic for small residuals, linear for large); "
        "less sensitive to outliers than OLS. epsilon=1.35, default alpha (L2 on coefs)."
    ),
}


class SkewedLog1pFeatures(BaseEstimator, TransformerMixin):
    """
    Apply log1p to heavily skewed columns (interaction-ready inputs).
    Skips columns with min <= -1 (e.g. Longitude) where log1p is undefined.
    """

    def __init__(self, skew_threshold: float = 0.5):
        self.skew_threshold = skew_threshold
        self.log_columns_: np.ndarray | None = None

    def fit(self, X: Any, y: Any = None) -> SkewedLog1pFeatures:
        X = np.asarray(X, dtype=float)
        n = X.shape[1]
        mask = np.zeros(n, dtype=bool)
        for j in range(n):
            col = X[:, j]
            if np.nanmin(col) <= -0.999:
                continue
            sk = abs(float(scipy_stats.skew(col, bias=False)))
            if sk > self.skew_threshold:
                mask[j] = True
        self.log_columns_ = mask
        return self

    def transform(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=float).copy()
        if self.log_columns_ is None:
            raise RuntimeError("SkewedLog1pFeatures is not fitted.")
        for j in range(X.shape[1]):
            if self.log_columns_[j]:
                X[:, j] = np.log1p(np.clip(X[:, j], a_min=-0.999999, a_max=None))
        return X


class TargetLogIfSkewed(BaseEstimator, TransformerMixin):
    """Fit on training target: optionally use log1p(y) when skew is high."""

    def __init__(self, skew_threshold: float = 0.25):
        self.skew_threshold = skew_threshold
        self.use_log_: bool = False

    def fit(self, y: Any) -> TargetLogIfSkewed:
        y = np.asarray(y, dtype=float).ravel()
        self.use_log_ = abs(float(scipy_stats.skew(y, bias=False))) > self.skew_threshold
        return self

    def transform(self, y: Any) -> np.ndarray:
        y = np.asarray(y, dtype=float).ravel()
        if self.use_log_:
            return np.log1p(np.clip(y, a_min=-0.999999, a_max=None))
        return y

    def inverse_transform(self, y: Any) -> np.ndarray:
        y = np.asarray(y, dtype=float).ravel()
        if self.use_log_:
            return np.expm1(y)
        return y


class TopKForestImportanceSelector(BaseEstimator, TransformerMixin):
    """Keep top-k features by RandomForestRegressor feature_importances_."""

    def __init__(self, k: int = 50, random_state: int = RANDOM_STATE):
        self.k = k
        self.random_state = random_state
        self.rf_: RandomForestRegressor | None = None
        self.support_: np.ndarray | None = None

    def fit(self, X: Any, y: Any) -> TopKForestImportanceSelector:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[1]
        k_eff = min(self.k, n)
        self.rf_ = RandomForestRegressor(
            n_estimators=120,
            max_depth=14,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.rf_.fit(X, y)
        imp = self.rf_.feature_importances_
        top = np.argsort(imp)[-k_eff:]
        self.support_ = np.zeros(n, dtype=bool)
        self.support_[top] = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.support_ is None:
            raise RuntimeError("TopKForestImportanceSelector is not fitted.")
        return X[:, self.support_]


def make_feature_preprocessor(
    selection: str = FE_SELECTION_METHOD,
    *,
    poly_degree: int = FE_POLY_DEGREE,
    n_top: int = FE_N_TOP_FEATURES,
    rfe_step: int = FE_RFE_STEP,
) -> Pipeline:
    """
    Pipeline: skew log -> PolynomialFeatures (degree includes 1..poly_degree) ->
    scale -> RFE(Ridge) or RF importance top-k.
    """
    steps: list[tuple[str, Any]] = [
        ("skew_log", SkewedLog1pFeatures(FE_SKEW_THRESHOLD)),
        ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
        ("scale_rfe", StandardScaler()),
    ]
    if selection == "rfe":
        steps.append(
            (
                "feat_select",
                RFE(
                    estimator=Ridge(alpha=1.0, random_state=RANDOM_STATE),
                    n_features_to_select=n_top,
                    step=rfe_step,
                ),
            )
        )
    elif selection == "rf":
        steps.append(
            ("feat_select", TopKForestImportanceSelector(k=n_top, random_state=RANDOM_STATE))
        )
    else:
        raise ValueError("selection must be 'rfe' or 'rf'")
    return Pipeline(steps)


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_features: int
) -> dict[str, float]:
    """RMSE, MAE, R², Adjusted R², MAPE (%), Explained Variance."""
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    denom = n - n_features - 1
    adj_r2 = np.nan if denom <= 0 else 1.0 - (1.0 - r2) * (n - 1) / denom
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2),
        "Adj_R2": float(adj_r2),
        "MAPE_pct": float(mean_absolute_percentage_error(y_true, y_pred) * 100.0),
        "Explained_Var": float(explained_variance_score(y_true, y_pred)),
    }


def _make_linear_ridge(*, use_fe: bool, fe_selection: str) -> Pipeline:
    alphas = np.logspace(-2, 4, 120)
    tail: list[tuple[str, Any]] = [
        ("scaler", StandardScaler()),
        (
            "est",
            RidgeCV(alphas=alphas, cv=5),
        ),
    ]
    if use_fe:
        return Pipeline([("fe", make_feature_preprocessor(fe_selection)), *tail])
    return Pipeline(tail)


def _make_linear_lasso(*, use_fe: bool, fe_selection: str) -> Pipeline:
    tail: list[tuple[str, Any]] = [
        ("scaler", StandardScaler()),
        (
            "est",
            LassoCV(
                cv=5,
                random_state=RANDOM_STATE,
                max_iter=20000,
            ),
        ),
    ]
    if use_fe:
        return Pipeline([("fe", make_feature_preprocessor(fe_selection)), *tail])
    return Pipeline(tail)


def _make_elastic_net(*, use_fe: bool, fe_selection: str) -> Pipeline:
    tail: list[tuple[str, Any]] = [
        ("scaler", StandardScaler()),
        (
            "est",
            ElasticNetCV(
                l1_ratio=[0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.0],
                cv=5,
                random_state=RANDOM_STATE,
                max_iter=20000,
            ),
        ),
    ]
    if use_fe:
        return Pipeline([("fe", make_feature_preprocessor(fe_selection)), *tail])
    return Pipeline(tail)


def _make_huber(*, use_fe: bool, fe_selection: str) -> Pipeline:
    tail: list[tuple[str, Any]] = [
        ("scaler", StandardScaler()),
        (
            "est",
            HuberRegressor(
                epsilon=1.35,
                max_iter=20000,
            ),
        ),
    ]
    if use_fe:
        return Pipeline([("fe", make_feature_preprocessor(fe_selection)), *tail])
    return Pipeline(tail)


def _make_xgb_random_search(*, use_fe: bool, fe_selection: str) -> Pipeline | RandomizedSearchCV:
    base = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    param_dist: dict[str, Any] = {
        "max_depth": randint(3, 11),
        "learning_rate": loguniform(1e-2, 3e-1),
        "n_estimators": randint(80, 401),
        "subsample": uniform(0.55, 0.45),
        "colsample_bytree": uniform(0.55, 0.45),
        "reg_lambda": loguniform(1e-3, 25.0),
        "reg_alpha": loguniform(1e-3, 25.0),
        "min_child_weight": loguniform(0.5, 20.0),
        "gamma": loguniform(1e-3, 2.0),
    }
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=12,
        cv=inner_cv,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    if use_fe:
        return Pipeline([("fe", make_feature_preprocessor(fe_selection)), ("search", search)])
    return search


def build_estimator_factories(
    *,
    use_fe: bool = USE_FEATURE_ENGINEERING,
    fe_selection: str = FE_SELECTION_METHOD,
) -> dict[str, Callable[[], Any]]:
    return {
        "Linear Regression": lambda: (
            Pipeline(
                [
                    ("fe", make_feature_preprocessor(fe_selection)),
                    ("scaler", StandardScaler()),
                    ("est", LinearRegression()),
                ]
            )
            if use_fe
            else Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("est", LinearRegression()),
                ]
            )
        ),
        "Ridge (RidgeCV)": lambda: _make_linear_ridge(use_fe=use_fe, fe_selection=fe_selection),
        "Lasso (LassoCV)": lambda: _make_linear_lasso(use_fe=use_fe, fe_selection=fe_selection),
        "Elastic Net (ElasticNetCV)": lambda: _make_elastic_net(
            use_fe=use_fe, fe_selection=fe_selection
        ),
        "XGBoost (tuned)": lambda: _make_xgb_random_search(
            use_fe=use_fe, fe_selection=fe_selection
        ),
        HUBER_REGRESSOR_NAME: lambda: _make_huber(use_fe=use_fe, fe_selection=fe_selection),
    }


def run_repeated_cv_models(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
    n_repeats: int = 3,
    use_feature_engineering: bool = USE_FEATURE_ENGINEERING,
    fe_selection: str = FE_SELECTION_METHOD,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, list[float]]]]:
    """
    Repeated K-fold CV: per fold fit model (with model-specific training loss),
    collect test metrics. XGBoost uses RandomizedSearchCV on train split only.
    With feature engineering: skew log, polynomial (deg 1..FE_POLY_DEGREE),
    RFE or RF top-k, optional log1p on skewed target; adjusted R2 uses selected K.
    """
    n_features_adj = FE_N_TOP_FEATURES if use_feature_engineering else X.shape[1]
    cv = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
    )
    factories = build_estimator_factories(
        use_fe=use_feature_engineering, fe_selection=fe_selection
    )
    metric_names = ["RMSE", "MAE", "R2", "Adj_R2", "MAPE_pct", "Explained_Var"]
    fold_records: list[dict[str, Any]] = []
    scores_by_model: dict[str, dict[str, list[float]]] = {
        name: {m: [] for m in metric_names} for name in factories
    }

    X_np = X.to_numpy(dtype=np.float64)
    y_np = y.to_numpy(dtype=np.float64)
    # California rows are geographically ordered; shuffle indices so folds are i.i.d.-like.
    rng = np.random.default_rng(RANDOM_STATE)
    order = rng.permutation(len(y_np))
    X_np, y_np = X_np[order], y_np[order]

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X_np, y_np), start=1):
        X_train, X_test = X_np[train_idx], X_np[test_idx]
        y_train, y_test = y_np[train_idx], y_np[test_idx]

        target_tf: TargetLogIfSkewed | None = None
        y_train_fit = y_train
        if use_feature_engineering and FE_TARGET_SKEW_LOG:
            target_tf = TargetLogIfSkewed(FE_TARGET_SKEW_THRESHOLD)
            target_tf.fit(y_train)
            y_train_fit = target_tf.transform(y_train)

        for name, factory in factories.items():
            est = factory()
            est.fit(X_train, y_train_fit)
            y_pred = est.predict(X_test)
            if target_tf is not None and target_tf.use_log_:
                y_pred = target_tf.inverse_transform(y_pred)
            if use_feature_engineering:
                # MedHouseVal in this dataset lies in ~[0.15, 5]; clip tames rare log-fit explosions.
                y_pred = np.clip(y_pred, 0.15, 5.0)
            m = regression_metrics(y_test, y_pred, n_features_adj)
            for k, v in m.items():
                scores_by_model[name][k].append(v)
            row = {"fold": fold_id, "model": name, **m}
            fold_records.append(row)

    long_df = pd.DataFrame(fold_records)
    summary_rows = []
    for name in factories:
        row: dict[str, Any] = {"model": name}
        for m in metric_names:
            vals = np.asarray(scores_by_model[name][m], dtype=float)
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"] = float(vals.std(ddof=0))
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    return summary_df, long_df, scores_by_model


def _fit_predict_numpy(
    factory: Callable[[], Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    use_feature_engineering: bool,
) -> np.ndarray:
    """Fit on train, predict test; same target log1p handling as repeated CV."""
    target_tf: TargetLogIfSkewed | None = None
    y_train_fit = y_train
    if use_feature_engineering and FE_TARGET_SKEW_LOG:
        target_tf = TargetLogIfSkewed(FE_TARGET_SKEW_THRESHOLD)
        target_tf.fit(y_train)
        y_train_fit = target_tf.transform(y_train)
    est = factory()
    est.fit(X_train, y_train_fit)
    y_pred = est.predict(X_test)
    if target_tf is not None and target_tf.use_log_:
        y_pred = target_tf.inverse_transform(y_pred)
    if use_feature_engineering:
        y_pred = np.clip(y_pred, 0.15, 5.0)
    return y_pred


def evaluate_holdout_once(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    use_feature_engineering: bool = USE_FEATURE_ENGINEERING,
    fe_selection: str = FE_SELECTION_METHOD,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    """
    Single evaluation on held-out test (no refitting on test).
    Returns metrics table, predictions per model, and y_test as numpy vector.
    """
    n_features_adj = FE_N_TOP_FEATURES if use_feature_engineering else X_train.shape[1]
    factories = build_estimator_factories(
        use_fe=use_feature_engineering, fe_selection=fe_selection
    )
    X_tr = X_train.to_numpy(dtype=np.float64)
    y_tr = y_train.to_numpy(dtype=np.float64).ravel()
    X_te = X_test.to_numpy(dtype=np.float64)
    y_te = y_test.to_numpy(dtype=np.float64).ravel()
    preds: dict[str, np.ndarray] = {}
    rows_list: list[dict[str, Any]] = []
    for name, fac in factories.items():
        y_pred = _fit_predict_numpy(
            fac, X_tr, y_tr, X_te, use_feature_engineering=use_feature_engineering
        )
        preds[name] = y_pred
        rows_list.append({"model": name, **regression_metrics(y_te, y_pred, n_features_adj)})
    return pd.DataFrame(rows_list), preds, y_te


def plot_residual_diagnostics_grid(
    y_true: np.ndarray,
    preds_by_model: dict[str, np.ndarray],
    model_top: str,
    model_bottom: str,
    out_path: Path,
) -> None:
    """2x2: fitted vs residuals and Q-Q for two models (e.g. CV-best vs Huber)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for row, model_name in enumerate((model_top, model_bottom)):
        y_hat = preds_by_model[model_name]
        resid = y_true - y_hat
        ax_fit = axes[row, 0]
        ax_qq = axes[row, 1]
        ax_fit.scatter(y_hat, resid, alpha=0.35, s=14, c="steelblue", edgecolors="none")
        ax_fit.axhline(0.0, color="black", lw=0.9, ls="--")
        ax_fit.set_xlabel("Fitted (predicted)")
        ax_fit.set_ylabel("Residual (y − ŷ)")
        ax_fit.set_title(f"{model_name} — fitted vs residual")
        scipy_stats.probplot(resid, dist="norm", plot=ax_qq)
        ax_qq.set_title(f"{model_name} — Q-Q of residuals (normality check)")
    fig.suptitle(
        "Hold-out set: residual diagnostics (homoscedasticity / normality)",
        y=1.02,
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def paired_model_tests(
    scores_by_model: dict[str, dict[str, list[float]]],
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Paired t-test and Wilcoxon signed-rank on the same CV folds (model_a - model_b).
    Bonferroni correction: alpha / K with K = number of model pairs per metric.
    """
    metric_names = next(iter(scores_by_model.values())).keys()
    names = list(scores_by_model.keys())
    n_pairs = len(names) * (len(names) - 1) // 2
    alpha_bonf = alpha / n_pairs
    rows: list[dict[str, Any]] = []

    for metric in metric_names:
        for i, j in combinations(range(len(names)), 2):
            mi, mj = names[i], names[j]
            a = np.asarray(scores_by_model[mi][metric], dtype=float)
            b = np.asarray(scores_by_model[mj][metric], dtype=float)
            if len(a) != len(b):
                raise ValueError("Paired score lists must match fold count.")
            diff = a - b
            if np.allclose(diff, 0.0, rtol=0.0, atol=1e-15):
                t_stat, p_t = 0.0, 1.0
                w_stat, p_w = 0.0, 1.0
            else:
                t_stat, p_t = ttest_rel(a, b)
                p_w = np.nan
                w_stat = np.nan
                try:
                    w_res = wilcoxon(
                        diff,
                        zero_method="wilcox",
                        alternative="two-sided",
                        mode="auto",
                    )
                    w_stat = float(w_res.statistic)
                    p_w = float(w_res.pvalue)
                except ValueError:
                    pass

            rows.append(
                {
                    "metric": metric,
                    "model_a": mi,
                    "model_b": mj,
                    "mean_diff_a_minus_b": float(np.mean(diff)),
                    "ttest_statistic": float(t_stat),
                    "ttest_pvalue": float(p_t),
                    "wilcoxon_statistic": w_stat,
                    "wilcoxon_pvalue": p_w,
                    "sig_ttest_uncorrected": p_t < alpha,
                    "sig_wilcoxon_uncorrected": (p_w < alpha)
                    if not np.isnan(p_w)
                    else False,
                    "bonferroni_alpha": alpha_bonf,
                    "sig_ttest_bonferroni": p_t < alpha_bonf,
                    "sig_wilcoxon_bonferroni": (p_w < alpha_bonf)
                    if not np.isnan(p_w)
                    else False,
                }
            )

    return pd.DataFrame(rows)


def print_paired_tests_short(tests_df: pd.DataFrame) -> None:
    """Print RMSE and R2 pairwise tables (full table in CSV)."""
    rmse_rows = tests_df[tests_df["metric"] == "RMSE"]
    n_pairs = len(rmse_rows)
    print("\n" + "=" * 72)
    print("Paired tests on CV folds (same folds -> paired)")
    print("H0: no difference in metric. mean_diff = model_a - model_b.")
    print("RMSE/MAE/MAPE: negative mean_diff => model_a better (lower).")
    print("R2 / Adj_R2 / Explained_Var: positive mean_diff => model_a better (higher).")
    print(f"Bonferroni: alpha / {n_pairs} model pairs per metric (K = C(n_models, 2)).")
    print("=" * 72)
    for metric in ("RMSE", "R2"):
        sub = tests_df[tests_df["metric"] == metric]
        if sub.empty:
            continue
        print(f"\n--- {metric} ---")
        cols = [
            "model_a",
            "model_b",
            "mean_diff_a_minus_b",
            "ttest_pvalue",
            "wilcoxon_pvalue",
            "sig_ttest_bonferroni",
            "sig_wilcoxon_bonferroni",
        ]
        print(sub[cols].to_string(index=False))


def load_california_housing() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load sklearn California Housing as a DataFrame and a target Series (MedHouseVal).
    """
    bunch = fetch_california_housing(as_frame=True)
    X = bunch.frame.drop(columns=["MedHouseVal"])
    y = bunch.frame["MedHouseVal"]
    return X, y


def dataframe_overview(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Combine features and target for EDA; return full DataFrame."""
    df = X.copy()
    df["MedHouseVal"] = y.values
    return df


def eda_feature_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    """Histograms / KDE for all numeric columns."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(df.columns)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, df.columns):
        sns.histplot(df[col], kde=True, ax=ax, color="steelblue", edgecolor="white")
        ax.set_title(col)
    for j in range(len(df.columns), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("California Housing — feature & target distributions", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "01_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def eda_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Pearson correlation matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def eda_pairwise_interactions(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Selected pairwise scatter plots (interaction / joint behaviour).
    """
    pairs: list[tuple[str, str, str]] = [
        ("MedInc", "AveRooms", "AveRooms vs MedInc"),
        ("AveBedrms", "AveRooms", "AveRooms vs AveBedrms"),
        ("Longitude", "Latitude", "Latitude vs Longitude"),
        ("MedInc", "MedHouseVal", "MedHouseVal vs MedInc"),
        ("HouseAge", "Population", "Population vs HouseAge"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes_flat = axes.ravel()
    for ax, (x_col, y_col, title) in zip(axes_flat, pairs):
        ax.scatter(
            df[x_col],
            df[y_col],
            alpha=0.25,
            s=6,
            c="steelblue",
            edgecolors="none",
        )
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
    axes_flat[-1].set_visible(False)
    fig.suptitle("California Housing — pairwise interactions", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "03_pairwise_interactions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def eda_outliers_iqr_zscore(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    Per-column outlier counts: IQR rule (1.5 * IQR) and |z| > 3.
    Returns summary table.
    """
    rows = []
    for col in df.columns:
        s = df[col].astype(float)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask_iqr = (s < low) | (s > high)
        z = (s - s.mean()) / s.std(ddof=0)
        mask_z = z.abs() > 3
        rows.append(
            {
                "column": col,
                "n_iqr": int(mask_iqr.sum()),
                "pct_iqr": 100.0 * mask_iqr.mean(),
                "n_zscore_gt3": int(mask_z.sum()),
                "pct_zscore": 100.0 * mask_z.mean(),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "04_outlier_summary.csv", index=False)
    return summary


def print_model_benchmark(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("MODELS -- training loss / objective (fit criterion)")
    print("=" * 72)
    for name, desc in MODEL_TRAINING_LOSS.items():
        print(f"\n- {name}")
        print(f"  {desc}")

    print("\n" + "=" * 72)
    print("MODELS -- test metrics: 5-fold CV repeated 3x (mean +/- std)")
    print("=" * 72)
    metric_cols = [c for c in summary.columns if c.endswith("_mean")]
    for _, row in summary.iterrows():
        print(f"\n{row['model']}")
        for c in metric_cols:
            mname = c.replace("_mean", "")
            mean = row[c]
            std = row[f"{mname}_std"]
            print(f"  {mname}: {mean:.6f} +/- {std:.6f}")


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_theme(font_scale=0.95)

    X, y = load_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=HOLDOUT_TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    df_train = dataframe_overview(X_train, y_train)

    print("California Housing")
    print("  full shape (rows, features):", X.shape)
    print(
        f"  train / hold-out test split: {len(y_train)} / {len(y_test)} "
        f"(test_size={HOLDOUT_TEST_SIZE}, shuffle=True, random_state={RANDOM_STATE})"
    )
    print("  columns:", list(X.columns))
    print("\nHead (train):")
    print(df_train.head())
    print("\nDescribe (train only — EDA / CV use this split; test not used until final block):")
    print(df_train.describe().T)

    print("\n--- EDA (training split only): saving figures to", FIG_DIR, "---")
    eda_feature_distributions(df_train, FIG_DIR)
    eda_correlation_heatmap(df_train, FIG_DIR)
    eda_pairwise_interactions(df_train, FIG_DIR)
    out_summary = eda_outliers_iqr_zscore(df_train, FIG_DIR)
    print("\nOutlier summary (IQR 1.5x and |z|>3):")
    print(out_summary.to_string(index=False))

    print("\n--- Model comparison (repeated CV on training split only) ---")
    if USE_FEATURE_ENGINEERING:
        print(
            "Feature engineering ON: skewed cols -> log1p (where valid); "
            f"PolynomialFeatures(degree={FE_POLY_DEGREE}, interactions included); "
            f"selection={FE_SELECTION_METHOD} (top {FE_N_TOP_FEATURES} features); "
            f"target log1p if |skew(y_train)| > {FE_TARGET_SKEW_THRESHOLD}."
        )
    summary_df, folds_df, scores_by_model = run_repeated_cv_models(
        X_train,
        y_train,
        n_splits=5,
        n_repeats=3,
        use_feature_engineering=USE_FEATURE_ENGINEERING,
        fe_selection=FE_SELECTION_METHOD,
    )
    summary_df.to_csv(RESULTS_CSV, index=False)
    folds_df.to_csv(RESULTS_FOLDS_CSV, index=False)
    print_model_benchmark(summary_df)
    print(f"\nSummary: {RESULTS_CSV}")
    print(f"Per-fold metrics: {RESULTS_FOLDS_CSV}")

    tests_df = paired_model_tests(scores_by_model, alpha=0.05)
    tests_df.to_csv(PAIRWISE_TESTS_CSV, index=False)
    print_paired_tests_short(tests_df)
    print(f"\nAll pairwise tests (all metrics): {PAIRWISE_TESTS_CSV}")

    print("\n" + "=" * 72)
    print(
        "FINAL hold-out evaluation (single use of test set; same preprocessing as CV)"
    )
    print("=" * 72)
    holdout_df, holdout_preds, y_holdout = evaluate_holdout_once(
        X_train,
        y_train,
        X_test,
        y_test,
        use_feature_engineering=USE_FEATURE_ENGINEERING,
        fe_selection=FE_SELECTION_METHOD,
    )
    holdout_df.to_csv(FINAL_HOLDOUT_METRICS_CSV, index=False)
    print(holdout_df.to_string(index=False))
    print(f"\nSaved: {FINAL_HOLDOUT_METRICS_CSV}")

    best_cv_row = summary_df.loc[summary_df["RMSE_mean"].idxmin()]
    best_cv_name = str(best_cv_row["model"])
    huber_name = HUBER_REGRESSOR_NAME
    if best_cv_name == huber_name:
        ordered = summary_df.sort_values("RMSE_mean")["model"].tolist()
        contrast_name = ordered[1] if len(ordered) > 1 else ordered[0]
        model_top, model_bottom = contrast_name, huber_name
        print(
            f"\nResidual diagnostics: CV-best RMSE is Huber; comparing "
            f"second-best CV ({model_top}) vs {huber_name}."
        )
    else:
        model_top, model_bottom = best_cv_name, huber_name
        print(
            f"\nResidual diagnostics: CV-best by mean RMSE = {best_cv_name}; "
            f"second row = robust {huber_name}."
        )
    resid_fig = FIG_DIR / "05_residual_diagnostics_holdout.png"
    plot_residual_diagnostics_grid(
        y_holdout,
        holdout_preds,
        model_top=model_top,
        model_bottom=model_bottom,
        out_path=resid_fig,
    )
    print(f"Residual / Q-Q figure: {resid_fig}")
    print("\nDone.")


if __name__ == "__main__":
    main()
