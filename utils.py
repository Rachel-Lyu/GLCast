import json
import os
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import celer
except Exception:  # pragma: no cover - optional dependency
    celer = None
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error


"""
This module contains the full runnable method implementation:

1) build state-specific lagged designs;
2) fit the exponentially weighted AR baseline;
3) fit the SIS + Group Lasso model with KKT checking;
4) rank transferable auxiliary states;
5) aggregate candidate-state predictions;
6) run the full walk-forward train/validation/test loop; and
7) save / print interpretable model-selection summaries.

Compared with your earlier snippet, the main addition is the training wrapper
`run_walkforward_training(...)`, which now owns the rolling loop and emits files
for predictions, aggregation candidates, and signed selected auxiliary signals.
"""


# -----------------------------------------------------------------------------
# Aggregation: candidate-state screening + Q-aggregation on validation
# -----------------------------------------------------------------------------
def estimate_R_hat_from_series(d0, dl, s: int) -> float:
    r"""Approximate the screened discrepancy \hat{R}_l used for candidate ranking."""
    if hasattr(d0, "index") and hasattr(dl, "index"):
        common = d0.index.intersection(dl.index)
        if len(common) == 0:
            return np.inf
        delta = (dl.loc[common] - d0.loc[common]).to_numpy()
        m = len(delta)
    else:
        d0 = np.asarray(d0, float).ravel()
        dl = np.asarray(dl, float).ravel()
        m = min(len(d0), len(dl))
        if m == 0:
            return np.inf
        delta = dl[:m] - d0[:m]

    s = int(min(s, m))
    if s <= 0:
        return 0.0
    idx = np.argpartition(np.abs(delta), -s)[-s:]
    return float(np.sum(delta[idx] ** 2))



def select_candidate_states(
    d_by_state: Dict[str, pd.Series],
    target_state: str,
    all_states: Iterable[str],
    s: int = 300,
    K: int = 10,
) -> List[str]:
    d0 = d_by_state[target_state]
    scores: List[Tuple[str, float]] = []
    for st in all_states:
        if st == target_state or st not in d_by_state:
            continue
        scores.append((st, estimate_R_hat_from_series(d0, d_by_state[st], s=s)))
    scores.sort(key=lambda x: x[1])
    return [st for st, _ in scores[:K]]



def q_aggregate_weights(
    y_val,
    Yhat_val,
    total_step: int = 10,
    selection: bool = False,
    eps: float = 1e-3,
    tau: float = 1.0,
):
    """
    Q-aggregation in prediction space.

    `tau` defaults to 1.0 so the behavior matches your current code, but keeping it
    explicit makes the training loop consistent with the `Q_TAU` hyperparameter you
    already store in the config block.
    """
    y = np.asarray(y_val, float).reshape(-1, 1)
    XB = np.asarray(Yhat_val, float)
    _, M = XB.shape

    if M == 0:
        return np.array([], dtype=float)
    if M == 1:
        return np.array([1.0], dtype=float)

    tau = float(max(tau, 1e-12))

    if selection:
        mse = np.mean((y - XB) ** 2, axis=0)
        khat = int(np.argmin(mse))
        theta_hat = np.zeros(M, dtype=float)
        theta_hat[khat] = 1.0
        return theta_hat

    rss = np.sum((y - XB) ** 2, axis=0)
    a = -0.5 * rss / tau
    a = a - np.max(a)
    theta_hat = np.exp(a)
    theta_hat /= theta_hat.sum()

    theta_old = theta_hat.copy()
    beta_pred = XB @ theta_hat

    for _ in range(int(total_step)):
        Xbeta = beta_pred.reshape(-1, 1)
        adj = np.sum((Xbeta - XB) ** 2, axis=0) / (8.0 * tau)
        a = -0.5 * rss / tau + adj
        a = a - np.max(a)
        theta_hat = np.exp(a)
        theta_hat /= theta_hat.sum()

        beta_new = (XB @ theta_hat) * 0.25 + 0.75 * beta_pred
        if np.sum(np.abs(theta_hat - theta_old)) < eps:
            beta_pred = beta_new
            break
        theta_old = theta_hat.copy()
        beta_pred = beta_new

    return theta_hat


# -----------------------------------------------------------------------------
# Dataset construction
# -----------------------------------------------------------------------------
def build_design_for_state(
    target_wide: pd.DataFrame,
    feature_wides: Dict[str, pd.DataFrame],
    state: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    feature_lags: Sequence[int],
    ar_lags: Sequence[int],
    unit: str,
):
    """Build one lagged design matrix for one state over the requested date span."""
    max_lag = max(max(feature_lags), max(ar_lags))
    start_ext = start_date - pd.Timedelta(max_lag, unit=unit)

    idx = target_wide.loc[start_ext:end_date].index
    if state not in target_wide.columns or len(idx) == 0:
        return None

    y_series = target_wide.loc[idx, state].copy()

    X_blocks: List[pd.Series] = []
    feature_names: List[str] = []
    group_names: List[str] = []

    for sig_name, wide in feature_wides.items():
        if state not in wide.columns:
            continue
        s = wide.loc[idx, state].copy()
        for lag in feature_lags:
            X_blocks.append(s.shift(lag))
            feature_names.append(f"{sig_name}_lag{lag}")
        group_names.append(sig_name)

    for lag in ar_lags:
        X_blocks.append(y_series.shift(lag))
        feature_names.append(f"AR_lag{lag}")
    group_names.append("AR")

    if len(X_blocks) == 0:
        return None

    X_df = pd.concat(X_blocks, axis=1)
    X_df.columns = feature_names

    aligned = pd.concat([y_series.rename("y"), X_df], axis=1)
    aligned = aligned.loc[start_date:end_date].dropna()
    if len(aligned) == 0:
        return None

    y = aligned["y"].to_numpy(dtype=float)
    X = aligned.drop(columns=["y"]).to_numpy(dtype=float)
    dates = pd.to_datetime(aligned.index)
    return X, y, dates, feature_names, group_names



def compute_walkforward_schedule(
    dates: Sequence[pd.Timestamp],
    train_size: int,
    test_size: int,
    retrain_every: int,
    unit: str = "D",
) -> List[pd.Timestamp]:
    """Return the rolling sequence of training end dates."""
    dates = pd.to_datetime(pd.Index(dates)).sort_values()
    start = dates.min() + pd.Timedelta(train_size, unit=unit)
    end = dates.max() - pd.Timedelta(test_size, unit=unit)

    if start >= end:
        return []

    start = dates[dates.get_indexer([start], method="nearest")[0]]
    train_ends: List[pd.Timestamp] = []
    cur = start
    while cur <= end:
        train_ends.append(cur)
        cur = cur + pd.Timedelta(retrain_every, unit=unit)
    return train_ends


# -----------------------------------------------------------------------------
# Group Lasso helpers
# -----------------------------------------------------------------------------
def extend_indices(g_indices: Sequence[int], group_size: int) -> np.ndarray:
    g_indices = np.asarray(g_indices, dtype=int)
    return (g_indices[:, None] * group_size + np.arange(group_size)[None, :]).ravel().astype(int)



def shrink_indices(v_indices: Sequence[int], group_size: int) -> np.ndarray:
    v_indices = np.asarray(v_indices, dtype=int)
    return np.unique(v_indices // group_size).astype(int)



def standardize_xy(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0, ddof=0)
    X_std[X_std == 0] = 1.0
    Xz = (X - X_mean) / X_std

    y_mean = y.mean()
    y_std = y.std(ddof=0)
    if y_std == 0:
        y_std = 1.0
    yz = (y - y_mean) / y_std
    return Xz, yz, X_mean, X_std, y_mean, y_std



def recover_coefficients(unit_coef, X_std, y_std):
    X_std = np.asarray(X_std, dtype=float)
    return (y_std / X_std) * np.asarray(unit_coef, dtype=float)



def recover_intercept(X, y, coef_original, X_mean=None, y_mean=None) -> float:
    if X_mean is None:
        X_mean = np.mean(X, axis=0)
    if y_mean is None:
        y_mean = np.mean(y)
    return float(y_mean - X_mean.dot(coef_original))



def SIS(X, y, nSel: int) -> np.ndarray:
    X = np.asarray(X)
    y = np.asarray(y)
    y_std = y.std()
    if y_std == 0:
        return np.array([], dtype=int)

    corrs = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        xj = X[:, j]
        sx = xj.std()
        if sx == 0:
            corrs[j] = 0.0
        else:
            corrs[j] = abs(np.cov(xj, y, bias=True)[0, 1] / (sx * y_std))

    order = np.argsort(corrs)[::-1]
    return order[:nSel].astype(int)



def _group_soft_threshold(beta_group: np.ndarray, lam: float) -> np.ndarray:
    norm = np.linalg.norm(beta_group, 2)
    if norm == 0 or norm <= lam:
        return np.zeros_like(beta_group)
    return (1.0 - lam / norm) * beta_group



def _fit_grpLasso_fista(Xz, yz, alpha: float, group_size: int, weights=None, max_iter: int = 3000, tol: float = 1e-6):
    """Simple equal-group-size Group Lasso solver used when `celer` is unavailable."""
    Xz = np.asarray(Xz, dtype=float)
    yz = np.asarray(yz, dtype=float)
    n, p = Xz.shape
    n_groups = p // group_size

    if weights is None:
        weights_arr = np.ones(n_groups, dtype=float)
    else:
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.size == 1:
            weights_arr = np.repeat(float(weights_arr), n_groups)
        if len(weights_arr) != n_groups:
            raise ValueError('weights must have one entry per group.')

    spectral = np.linalg.norm(Xz, ord=2)
    L = (spectral ** 2) / max(n, 1)
    L = max(L, 1e-8)
    step = 1.0 / L

    beta = np.zeros(p, dtype=float)
    z = beta.copy()
    t = 1.0

    for _ in range(max_iter):
        grad = Xz.T @ (Xz @ z - yz) / max(n, 1)
        beta_next = z - step * grad

        for g in range(n_groups):
            j0 = g * group_size
            j1 = j0 + group_size
            beta_next[j0:j1] = _group_soft_threshold(beta_next[j0:j1], step * alpha * weights_arr[g])

        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = beta_next + ((t - 1.0) / t_next) * (beta_next - beta)

        if np.linalg.norm(beta_next - beta) <= tol * max(1.0, np.linalg.norm(beta)):
            beta = beta_next
            break

        beta = beta_next
        t = t_next

    return beta



def fit_grpLasso(Xz, yz, alpha: float, group_size: int, weights=None):
    if celer is not None:
        model = celer.GroupLasso(
            groups=group_size,
            alpha=alpha,
            tol=1e-4,
            fit_intercept=False,
            weights=weights,
        )
        model.fit(Xz, yz)
        return model, model.coef_.copy()

    coef = _fit_grpLasso_fista(Xz, yz, alpha=alpha, group_size=group_size, weights=weights)
    return None, coef



def check_KKT(Xz, yz, coef_unit_full, active_g_indices, group_size, alpha, weights=None, tol=1e-4):
    residual = yz - Xz.dot(coef_unit_full)
    p_groups = Xz.shape[1] // group_size
    stats = np.zeros(p_groups)

    for g in range(p_groups):
        j0 = g * group_size
        j1 = j0 + group_size
        Xg = Xz[:, j0:j1]
        stats[g] = np.linalg.norm(Xg.T.dot(residual), 2) / Xg.shape[0]

    thresh = alpha if weights is None else alpha * np.asarray(weights)
    violated = np.where(stats > (thresh + tol))[0]
    violated = list(set(violated) - set(np.asarray(active_g_indices, dtype=int)))
    violated.sort()
    return violated



def SIS_grpLasso_KKT(
    X_expanded,
    y,
    group_size: int,
    alpha: float,
    weights=None,
    sis_ratio: float = 1.0,
    max_iter: int = 50,
):
    n, p_exp = X_expanded.shape
    p_groups = p_exp // group_size

    Xz, yz, X_mean, X_std, y_mean, y_std = standardize_xy(X_expanded, y)
    coef_unit_full = np.zeros(p_exp, dtype=float)

    X3 = Xz.reshape(n, p_groups, group_size)
    X_group = np.sqrt(np.mean(X3 ** 2, axis=2))

    nSel = int(max(1, min(p_groups, np.floor((len(y) / group_size - 1) * sis_ratio))))
    violated_g = list(SIS(X_group, yz, nSel=nSel))

    if len(violated_g) == 0:
        coef = recover_coefficients(coef_unit_full, X_std, y_std)
        intercept = float(y_mean)
        return np.array([], dtype=int), coef, intercept, coef_unit_full.copy()

    active_g = np.array([], dtype=int)
    it = 0
    while len(violated_g) > 0:
        it += 1
        if it > max_iter:
            break

        active_g = np.array(sorted(set(active_g) | set(violated_g)), dtype=int)
        SIS_v = extend_indices(active_g, group_size)

        _, coef_unit_sub = fit_grpLasso(Xz[:, SIS_v], yz, alpha=alpha, group_size=group_size, weights=weights)
        coef_unit_full[:] = 0.0
        coef_unit_full[SIS_v] = coef_unit_sub

        violated_g = check_KKT(Xz, yz, coef_unit_full, active_g, group_size, alpha, weights=weights)

    coef = recover_coefficients(coef_unit_full, X_std, y_std)
    intercept = recover_intercept(X_expanded, y, coef, X_mean=X_mean, y_mean=y_mean)
    active_v = np.where(coef != 0)[0]
    active_g_final = shrink_indices(active_v, group_size)
    return active_g_final, coef, intercept, coef_unit_full.copy()


# -----------------------------------------------------------------------------
# Baseline: AR-only with exponentially decayed weights
# -----------------------------------------------------------------------------
def _time_age_in_units(train_dates: Sequence[pd.Timestamp], train_end: pd.Timestamp, unit: str) -> np.ndarray:
    td = pd.to_datetime(train_end) - pd.to_datetime(pd.Index(train_dates))
    unit = unit.upper()
    if unit.startswith("W"):
        return (td.days.astype(float) / 7.0).to_numpy()
    return td.days.astype(float).to_numpy()



def weighted_ols_fit_predict(X_train, y_train, w_train, X_pred):
    X_train = np.asarray(X_train, float)
    y_train = np.asarray(y_train, float)
    w_train = np.asarray(w_train, float)
    X_pred = np.asarray(X_pred, float)

    Xtr = np.column_stack([np.ones(len(X_train)), X_train])
    Xpr = np.column_stack([np.ones(len(X_pred)), X_pred])

    sw = np.sqrt(np.clip(w_train, 0.0, np.inf))
    Xw = Xtr * sw[:, None]
    yw = y_train * sw

    eps = 1e-8
    A = Xw.T @ Xw + eps * np.eye(Xw.shape[1])
    b = Xw.T @ yw
    beta = np.linalg.solve(A, b)
    return Xpr @ beta



def fit_ar_baseline_decay(X_train, y_train, X_val, y_val, train_dates, train_end, gammas, unit: str = "D") -> float:
    backcast_lag = _time_age_in_units(train_dates, train_end, unit=unit)
    best = None
    for g in gammas:
        w = np.exp(-g * backcast_lag)
        w = w / (w.max() if w.max() > 0 else 1.0)
        yhat_val = weighted_ols_fit_predict(X_train, y_train, w, X_val)
        mse = mean_squared_error(y_val, yhat_val)
        if best is None or mse < best["mse"]:
            best = {"gamma": g, "mse": mse}
    return float(best["gamma"])



def predict_named(X, feat_names_ref: Sequence[str], coef_series: pd.Series, intercept: float) -> np.ndarray:
    w = coef_series.reindex(feat_names_ref).fillna(0.0).to_numpy()
    return np.asarray(X) @ w + intercept


# -----------------------------------------------------------------------------
# Model-summary helpers
# -----------------------------------------------------------------------------
def _sign_label(value: float, tol: float = 1e-8) -> str:
    if value > tol:
        return "+"
    if value < -tol:
        return "-"
    return "0"



def _signal_name_from_feature(feature_name: str) -> str:
    if feature_name.startswith("AR_lag"):
        return "AR"
    if "_lag" in feature_name:
        return feature_name.rsplit("_lag", 1)[0]
    return feature_name



def _extract_selection_rows(
    coef: np.ndarray,
    coef_raw: np.ndarray,
    feat_names: Sequence[str],
    dataset_name: str,
    target_name: str,
    target_state: str,
    train_end: pd.Timestamp,
    alpha: float,
    coef_tol: float,
) -> Tuple[List[dict], List[dict]]:
    coef_series = pd.Series(np.asarray(coef, dtype=float), index=list(feat_names))
    coef_raw_series = pd.Series(np.asarray(coef_raw, dtype=float), index=list(feat_names))
    nz = coef_series[np.abs(coef_series) > coef_tol].sort_values(key=np.abs, ascending=False)

    feature_rows: List[dict] = []
    signal_rows: List[dict] = []

    for rank_idx, (feature_name, value) in enumerate(nz.items(), start=1):
        raw_value = float(coef_raw_series.loc[feature_name])
        signal_name = _signal_name_from_feature(feature_name)
        feature_rows.append(
            {
                "dataset": dataset_name,
                "target_signal": target_name,
                "target_state": target_state,
                "train_end": pd.to_datetime(train_end),
                "alpha": float(alpha),
                "feature_name": feature_name,
                "signal_name": signal_name,
                "is_auxiliary": signal_name != "AR",
                "coef": float(value),
                "abs_coef": float(abs(value)),
                "coef_raw": raw_value,
                "abs_coef_raw": float(abs(raw_value)),
                "sign": _sign_label(float(value), tol=coef_tol),
                "rank_abs_within_model": rank_idx,
            }
        )

    if len(nz) == 0:
        return feature_rows, signal_rows

    tmp = pd.DataFrame(feature_rows)
    grouped = (
        tmp.groupby(["signal_name", "is_auxiliary"], as_index=False)
        .agg(
            coef_sum=("coef", "sum"),
            abs_coef_sum=("abs_coef", "sum"),
            coef_raw_sum=("coef_raw", "sum"),
            abs_coef_raw_sum=("abs_coef_raw", "sum"),
            n_selected_features=("feature_name", "count"),
        )
        .sort_values("abs_coef_sum", ascending=False)
        .reset_index(drop=True)
    )
    grouped["rank_abs_within_model"] = np.arange(1, len(grouped) + 1)
    grouped["sign"] = grouped["coef_sum"].apply(lambda x: _sign_label(float(x), tol=coef_tol))

    for _, row in grouped.iterrows():
        signal_rows.append(
            {
                "dataset": dataset_name,
                "target_signal": target_name,
                "target_state": target_state,
                "train_end": pd.to_datetime(train_end),
                "alpha": float(alpha),
                "signal_name": row["signal_name"],
                "is_auxiliary": bool(row["is_auxiliary"]),
                "coef_sum": float(row["coef_sum"]),
                "abs_coef_sum": float(row["abs_coef_sum"]),
                "coef_raw_sum": float(row["coef_raw_sum"]),
                "abs_coef_raw_sum": float(row["abs_coef_raw_sum"]),
                "sign": row["sign"],
                "n_selected_features": int(row["n_selected_features"]),
                "rank_abs_within_model": int(row["rank_abs_within_model"]),
            }
        )

    return feature_rows, signal_rows



def _summarize_candidate_states(agg_df: pd.DataFrame) -> pd.DataFrame:
    if agg_df.empty:
        return pd.DataFrame()
    cand_only = agg_df.loc[~agg_df["is_self"]].copy()
    if cand_only.empty:
        return pd.DataFrame()

    summary = (
        cand_only.groupby(["dataset", "target_signal", "target_state", "candidate_state"], as_index=False)
        .agg(
            selection_count=("candidate_state", "size"),
            mean_weight=("weight", "mean"),
            median_weight=("weight", "median"),
            mean_rank=("rank", "mean"),
        )
        .sort_values(["target_state", "selection_count", "mean_weight"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    summary["rank_within_target_state"] = (
        summary.groupby(["dataset", "target_signal", "target_state"])["selection_count"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return summary



def _summarize_signed_aux_signals(selected_signal_df: pd.DataFrame) -> pd.DataFrame:
    if selected_signal_df.empty:
        return pd.DataFrame()
    aux = selected_signal_df.loc[selected_signal_df["is_auxiliary"]].copy()
    if aux.empty:
        return pd.DataFrame()

    summary = (
        aux.groupby(["dataset", "target_signal", "signal_name", "sign"], as_index=False)
        .agg(
            selection_count=("signal_name", "size"),
            mean_abs_coef_sum=("abs_coef_sum", "mean"),
            median_abs_coef_sum=("abs_coef_sum", "median"),
            n_states=("target_state", "nunique"),
            n_train_ends=("train_end", "nunique"),
        )
        .sort_values(["selection_count", "mean_abs_coef_sum"], ascending=[False, False])
        .reset_index(drop=True)
    )
    summary["global_rank"] = np.arange(1, len(summary) + 1)
    return summary



def _print_model_selection_summary(
    selected_feature_rows: List[dict],
    selected_signal_rows: List[dict],
    dataset_name: str,
    alpha: float,
    print_top_k: int,
) -> None:
    if selected_feature_rows:
        feat_df = pd.DataFrame(selected_feature_rows)
        feat_df = feat_df.sort_values(
            ["train_end", "target_state", "rank_abs_within_model"], ascending=[True, True, True]
        )
        print(f"\n=== Selected lagged features | dataset={dataset_name} | alpha={alpha:.4g} ===")
        for (train_end, st), g in feat_df.groupby(["train_end", "target_state"]):
            top = g.head(print_top_k)[["feature_name", "coef", "sign", "rank_abs_within_model"]]
            print(f"\ntrain_end={pd.to_datetime(train_end).date()}  state={st}")
            print(top.to_string(index=False))

    if selected_signal_rows:
        sig_df = pd.DataFrame(selected_signal_rows)
        sig_df = sig_df.sort_values(
            ["train_end", "target_state", "rank_abs_within_model"], ascending=[True, True, True]
        )
        print(f"\n=== Selected signals | dataset={dataset_name} | alpha={alpha:.4g} ===")
        for (train_end, st), g in sig_df.groupby(["train_end", "target_state"]):
            top = g.head(print_top_k)[["signal_name", "coef_sum", "sign", "rank_abs_within_model"]]
            print(f"\ntrain_end={pd.to_datetime(train_end).date()}  state={st}")
            print(top.to_string(index=False))


# -----------------------------------------------------------------------------
# Main training function
# -----------------------------------------------------------------------------
def run_walkforward_training(
    config,
    gl_alphas: Sequence[float] = (0.03, 0.05, 0.07, 0.1),
    output_dir: str = "outputs",
    coef_tol: float = 1e-8,
    print_top_k: int = 10,
    verbose: bool = True,
):
    """
    Run the full rolling train/validation/test loop for one prepared dataset.

    Saved files per alpha
    ---------------------
    - predictions.csv: y_true and y_pred with timestamps for each model
    - aggregation_weights.csv: candidate-state weights from Q-aggregation
    - selected_features.csv: non-negligible lag-level coefficients with signs and ranks
    - selected_signals.csv: signal-level signed summaries per state/window
    - most_frequent_candidate_states.csv: frequency-ranked aggregation candidates
    - most_frequent_signed_aux_signals.csv: frequency-ranked signed auxiliary signals
    - metrics_overall.csv / metrics_by_state.csv / rmse_by_step.csv
    - run_metadata.json: run settings for reproducibility
    """
    os.makedirs(output_dir, exist_ok=True)

    target_wide = config.target_wide.sort_index().copy()
    feature_wides = config.feature_wides
    states = sorted(st for st in target_wide.columns if st in set(config.valid_states))
    if len(states) == 0:
        states = sorted(target_wide.columns.tolist())

    feature_lags = tuple(config.feature_lags)
    ar_lags = tuple(config.ar_lags)
    group_size = len(feature_lags)
    if len(ar_lags) != group_size:
        raise ValueError("AR_LAGS must have the same length as FEATURE_LAGS for equal-size groups.")

    train_ends = compute_walkforward_schedule(
        dates=target_wide.index,
        train_size=config.train_window,
        test_size=config.test_window,
        retrain_every=config.retrain_every,
        unit=config.unit,
    )

    results_by_alpha = {}
    for alpha in gl_alphas:
        alpha_dir = os.path.join(output_dir, config.dataset_name, f"alpha_{alpha:.4g}")
        os.makedirs(alpha_dir, exist_ok=True)

        pred_rows: List[dict] = []
        agg_rows: List[dict] = []
        selected_feature_rows: List[dict] = []
        selected_signal_rows: List[dict] = []

        for train_end in train_ends:
            train_end = pd.to_datetime(train_end)
            train_start = train_end - pd.Timedelta(config.train_window - 1, unit=config.unit)
            val_start = train_end - pd.Timedelta(config.val_window - 1, unit=config.unit)
            test_start = train_end + pd.Timedelta(1, unit=config.unit)
            test_end = train_end + pd.Timedelta(config.test_window, unit=config.unit)

            fit_by_state = {}
            d_by_state = {}
            baseline_by_state = {}
            data_cache = {}

            for st in states:
                built = build_design_for_state(
                    target_wide=target_wide,
                    feature_wides=feature_wides,
                    state=st,
                    start_date=train_start,
                    end_date=test_end,
                    feature_lags=feature_lags,
                    ar_lags=ar_lags,
                    unit=config.unit,
                )
                if built is None:
                    continue

                X_all, y_all, dates_all, feat_names, group_names = built
                data_cache[st] = (X_all, y_all, dates_all, feat_names, group_names)

                dates_all = pd.to_datetime(dates_all)
                idx_train = (dates_all >= train_start) & (dates_all <= train_end)
                idx_val = (dates_all >= val_start) & (dates_all <= train_end)
                idx_test = (dates_all >= test_start) & (dates_all <= test_end)

                if idx_train.sum() < 10 or idx_val.sum() < 2 or idx_test.sum() < 2:
                    continue

                X_tr, y_tr, d_tr = X_all[idx_train], y_all[idx_train], dates_all[idx_train]
                X_va, y_va = X_all[idx_val], y_all[idx_val]

                ar_cols = np.arange(X_all.shape[1] - len(ar_lags), X_all.shape[1])
                Xtr_ar, Xva_ar = X_tr[:, ar_cols], X_va[:, ar_cols]
                best_gamma = fit_ar_baseline_decay(
                    Xtr_ar,
                    y_tr,
                    Xva_ar,
                    y_va,
                    train_dates=d_tr,
                    train_end=train_end,
                    gammas=config.baseline_gammas,
                    unit=config.unit,
                )
                baseline_by_state[st] = {"gamma": best_gamma}

                active_g, coef, intercept, coef_raw = SIS_grpLasso_KKT(
                    X_expanded=X_tr,
                    y=y_tr,
                    group_size=group_size,
                    alpha=float(alpha),
                    weights=None,
                )
                fit_by_state[st] = {
                    "coef": pd.Series(coef, index=feat_names),
                    "intercept": float(intercept),
                    "active_g": np.asarray(active_g, dtype=int),
                }
                d_by_state[st] = pd.Series((X_tr.T @ y_tr) / max(len(y_tr), 1), index=feat_names)

                feat_rows, sig_rows = _extract_selection_rows(
                    coef=coef,
                    coef_raw=coef_raw,
                    feat_names=feat_names,
                    dataset_name=config.dataset_name,
                    target_name=config.target_name,
                    target_state=st,
                    train_end=train_end,
                    alpha=float(alpha),
                    coef_tol=coef_tol,
                )
                selected_feature_rows.extend(feat_rows)
                selected_signal_rows.extend(sig_rows)

            if len(fit_by_state) < 2:
                continue

            for st in list(fit_by_state.keys()):
                X_all, y_all, dates_all, feat_names, group_names = data_cache[st]
                dates_all = pd.to_datetime(dates_all)

                idx_train = (dates_all >= train_start) & (dates_all <= train_end)
                idx_val = (dates_all >= val_start) & (dates_all <= train_end)
                idx_test = (dates_all >= test_start) & (dates_all <= test_end)
                if idx_train.sum() < 10 or idx_val.sum() < 2 or idx_test.sum() < 2:
                    continue

                X_tr, y_tr, d_tr = X_all[idx_train], y_all[idx_train], dates_all[idx_train]
                X_va, y_va = X_all[idx_val], y_all[idx_val]
                X_te, y_te, d_te = X_all[idx_test], y_all[idx_test], dates_all[idx_test]

                ar_cols = np.arange(X_all.shape[1] - len(ar_lags), X_all.shape[1])
                Xtr_ar, Xte_ar = X_tr[:, ar_cols], X_te[:, ar_cols]
                gamma = baseline_by_state[st]["gamma"]
                backcast_lag = _time_age_in_units(d_tr, train_end, unit=config.unit)
                w = np.exp(-gamma * backcast_lag)
                w = w / (w.max() if w.max() > 0 else 1.0)
                yhat_base = weighted_ols_fit_predict(Xtr_ar, y_tr, w, Xte_ar)

                fit0 = fit_by_state[st]
                yhat_gl = predict_named(X_te, feat_names, fit0["coef"], fit0["intercept"])

                all_states_here = [
                    s for s in fit_by_state.keys() if s in d_by_state and s in set(config.valid_states)
                ]
                cand_states = select_candidate_states(
                    d_by_state=d_by_state,
                    target_state=st,
                    all_states=all_states_here,
                    s=int(config.s_screen),
                    K=int(config.k_candidates),
                )
                cand_all = [st] + cand_states

                Yhat_val = []
                Yhat_test = []
                for s2 in cand_all:
                    fit2 = fit_by_state[s2]
                    Yhat_val.append(predict_named(X_va, feat_names, fit2["coef"], fit2["intercept"]))
                    Yhat_test.append(predict_named(X_te, feat_names, fit2["coef"], fit2["intercept"]))
                Yhat_val = np.column_stack(Yhat_val)
                Yhat_test = np.column_stack(Yhat_test)

                w_agg = q_aggregate_weights(
                    y_va,
                    Yhat_val,
                    total_step=10,
                    selection=False,
                    eps=1e-3,
                    tau=float(config.q_tau),
                )
                yhat_agg = Yhat_test @ w_agg

                for j, s2 in enumerate(cand_all):
                    agg_rows.append(
                        {
                            "dataset": config.dataset_name,
                            "target_signal": config.target_name,
                            "target_state": st,
                            "train_end": train_end,
                            "candidate_state": s2,
                            "is_self": bool(j == 0),
                            "rank": int(j),
                            "weight": float(w_agg[j]),
                            "alpha": float(alpha),
                        }
                    )

                for i, dt in enumerate(d_te):
                    pred_rows.extend(
                        [
                            {
                                "dataset": config.dataset_name,
                                "target_signal": config.target_name,
                                "geo_value": st,
                                "train_end": train_end,
                                "time_value": pd.to_datetime(dt),
                                "model": "baseline_AR_decay",
                                "y_true": float(y_te[i]),
                                "y_pred": float(yhat_base[i]),
                                "alpha": float(alpha),
                            },
                            {
                                "dataset": config.dataset_name,
                                "target_signal": config.target_name,
                                "geo_value": st,
                                "train_end": train_end,
                                "time_value": pd.to_datetime(dt),
                                "model": "SIS_GroupLasso",
                                "y_true": float(y_te[i]),
                                "y_pred": float(yhat_gl[i]),
                                "alpha": float(alpha),
                            },
                            {
                                "dataset": config.dataset_name,
                                "target_signal": config.target_name,
                                "geo_value": st,
                                "train_end": train_end,
                                "time_value": pd.to_datetime(dt),
                                "model": "SIS_GroupLasso_Agg",
                                "y_true": float(y_te[i]),
                                "y_pred": float(yhat_agg[i]),
                                "alpha": float(alpha),
                            },
                        ]
                    )

        pred_df = pd.DataFrame(pred_rows)
        agg_df = pd.DataFrame(agg_rows)
        selected_feature_df = pd.DataFrame(selected_feature_rows)
        selected_signal_df = pd.DataFrame(selected_signal_rows)

        metrics_by_state = pd.DataFrame()
        metrics_overall = pd.DataFrame()
        rmse_by_step = pd.DataFrame()
        candidate_summary = pd.DataFrame()
        signed_aux_summary = pd.DataFrame()

        if not pred_df.empty:
            pred_df["train_end"] = pd.to_datetime(pred_df["train_end"])
            pred_df["time_value"] = pd.to_datetime(pred_df["time_value"])

            metrics_by_state = (
                pred_df.groupby(["dataset", "target_signal", "model", "geo_value"], as_index=False)
                .apply(
                    lambda g: pd.Series(
                        {
                            "RMSE": float(np.sqrt(mean_squared_error(g["y_true"], g["y_pred"]))),
                            "MAE": float(mean_absolute_error(g["y_true"], g["y_pred"])),
                            "N": int(len(g)),
                        }
                    )
                )
                .reset_index(drop=True)
            )

            metrics_overall = (
                pred_df.groupby(["dataset", "target_signal", "model"], as_index=False)
                .apply(
                    lambda g: pd.Series(
                        {
                            "RMSE": float(np.sqrt(mean_squared_error(g["y_true"], g["y_pred"]))),
                            "MAE": float(mean_absolute_error(g["y_true"], g["y_pred"])),
                            "N": int(len(g)),
                        }
                    )
                )
                .reset_index(drop=True)
            )

            rmse_by_step = (
                pred_df.groupby(["dataset", "target_signal", "model", "train_end"], as_index=False)
                .apply(lambda g: pd.Series({"RMSE": float(np.sqrt(np.mean((g["y_true"] - g["y_pred"]) ** 2)))}))
                .reset_index(drop=True)
                .sort_values(["dataset", "target_signal", "model", "train_end"])
            )

        if not agg_df.empty:
            agg_df["train_end"] = pd.to_datetime(agg_df["train_end"])
            candidate_summary = _summarize_candidate_states(agg_df)

        if not selected_feature_df.empty:
            selected_feature_df["train_end"] = pd.to_datetime(selected_feature_df["train_end"])
        if not selected_signal_df.empty:
            selected_signal_df["train_end"] = pd.to_datetime(selected_signal_df["train_end"])
            signed_aux_summary = _summarize_signed_aux_signals(selected_signal_df)

        if verbose:
            if not metrics_overall.empty:
                print(f"\n### Overall metrics | dataset={config.dataset_name} | alpha={alpha:.4g}")
                print(metrics_overall.to_string(index=False))
            _print_model_selection_summary(
                selected_feature_rows=selected_feature_rows,
                selected_signal_rows=selected_signal_rows,
                dataset_name=config.dataset_name,
                alpha=float(alpha),
                print_top_k=print_top_k,
            )
            if not candidate_summary.empty:
                print(f"\n### Most frequent candidate states | dataset={config.dataset_name} | alpha={alpha:.4g}")
                print(candidate_summary.head(print_top_k).to_string(index=False))
            if not signed_aux_summary.empty:
                print(f"\n### Most frequent signed auxiliary signals | dataset={config.dataset_name} | alpha={alpha:.4g}")
                print(signed_aux_summary.head(print_top_k).to_string(index=False))

        pred_df.to_csv(os.path.join(alpha_dir, "predictions.csv"), index=False)
        agg_df.to_csv(os.path.join(alpha_dir, "aggregation_weights.csv"), index=False)
        selected_feature_df.to_csv(os.path.join(alpha_dir, "selected_features.csv"), index=False)
        selected_signal_df.to_csv(os.path.join(alpha_dir, "selected_signals.csv"), index=False)
        candidate_summary.to_csv(os.path.join(alpha_dir, "most_frequent_candidate_states.csv"), index=False)
        signed_aux_summary.to_csv(os.path.join(alpha_dir, "most_frequent_signed_aux_signals.csv"), index=False)
        metrics_by_state.to_csv(os.path.join(alpha_dir, "metrics_by_state.csv"), index=False)
        metrics_overall.to_csv(os.path.join(alpha_dir, "metrics_overall.csv"), index=False)
        rmse_by_step.to_csv(os.path.join(alpha_dir, "rmse_by_step.csv"), index=False)

        metadata = {
            "dataset": config.dataset_name,
            "target_name": config.target_name,
            "alpha": float(alpha),
            "n_states": int(len(states)),
            "n_train_ends": int(len(train_ends)),
            "feature_lags": list(feature_lags),
            "ar_lags": list(ar_lags),
            "baseline_gammas": list(config.baseline_gammas),
            "k_candidates": int(config.k_candidates),
            "s_screen": int(config.s_screen),
            "q_tau": float(config.q_tau),
            "unit": config.unit,
            "retrain_every": int(config.retrain_every),
            "test_window": int(config.test_window),
            "val_window": int(config.val_window),
            "train_window": int(config.train_window),
            "forecast_horizon": int(config.forecast_horizon),
        }
        with open(os.path.join(alpha_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        results_by_alpha[float(alpha)] = {
            "predictions": pred_df,
            "aggregation_weights": agg_df,
            "selected_features": selected_feature_df,
            "selected_signals": selected_signal_df,
            "candidate_summary": candidate_summary,
            "signed_aux_summary": signed_aux_summary,
            "metrics_by_state": metrics_by_state,
            "metrics_overall": metrics_overall,
            "rmse_by_step": rmse_by_step,
            "output_dir": alpha_dir,
        }

    return results_by_alpha
