import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

for GL_ALPHA in [0.1, 0.5, 1, 5]:
    pred_rows = []
    agg_rows = []
    for train_end in train_ends:
        train_end = pd.to_datetime(train_end)
        train_start = train_end - pd.Timedelta(TRAIN_WINDOW - 1, unit = unit)
        val_start = train_end - pd.Timedelta(VAL_WINDOW - 1, unit = unit)
        test_start = train_end + pd.Timedelta(1, unit = unit)
        test_end = train_end + pd.Timedelta(TEST_WINDOW, unit = unit)

        # Fit GroupLasso for ALL states in this window (target-only training per state)
        # Store coefs/intercepts and d = X^T y / T for candidate selection
        fit_by_state = {}
        d_by_state = {}

        # also store AR-only baseline (gamma chosen per state)
        baseline_by_state = {}

        # Pre-build datasets per state (train/val/test)
        data_cache = {}

        for st in states:
            # build full window design up to test_end so we can slice train/val/test cleanly
            built = build_design_for_state(
                target_wide, feature_wides, st,
                start_date=train_start, end_date=test_end,
                feature_lags=FEATURE_LAGS, ar_lags=AR_LAGS, unit = unit
            )
            if built is None:
                continue
            X_all, y_all, dates_all, feat_names, group_names = built
            data_cache[st] = (X_all, y_all, dates_all, feat_names, group_names)

            # indices for train/val/test within aligned dates_all
            dates_all = pd.to_datetime(dates_all)
            idx_train = (dates_all >= train_start) & (dates_all <= train_end)
            idx_val   = (dates_all >= val_start) & (dates_all <= train_end)
            idx_test  = (dates_all >= test_start) & (dates_all <= test_end)

            if idx_train.sum() < 10 or idx_val.sum() < 2 or idx_test.sum() < 2:
                continue

            X_tr, y_tr, d_tr = X_all[idx_train], y_all[idx_train], dates_all[idx_train]
            X_va, y_va = X_all[idx_val], y_all[idx_val]
            X_te, y_te, d_te = X_all[idx_test], y_all[idx_test], dates_all[idx_test]

            # ---------- Baseline: AR-only with decay gamma tuned on val ----------
            # AR columns are last len(AR_LAGS) columns by construction
            ar_cols = np.arange(X_all.shape[1] - len(AR_LAGS), X_all.shape[1])
            Xtr_ar, Xva_ar, Xte_ar = X_tr[:, ar_cols], X_va[:, ar_cols], X_te[:, ar_cols]
            best_gamma = fit_ar_baseline_decay(
                Xtr_ar, y_tr, Xva_ar, y_va, train_dates=d_tr, train_end=train_end, gammas=BASELINE_GAMMAS
            )
            baseline_by_state[st] = {"gamma": best_gamma}

            # ---------- SIS-GroupLasso fit ----------
            active_g, coef, intercept = SIS_grpLasso_KKT(
                X_expanded=X_tr, y=y_tr,
                group_size=group_size, alpha=GL_ALPHA, weights=None
            )
            fit_by_state[st] = {
                "coef": pd.Series(coef, index=feat_names),
                "intercept": intercept,
                "active_g": active_g
            }

            # d statistic for candidate selection (use training)
            d_by_state[st] = (X_tr.T @ y_tr) / max(len(y_tr), 1)

        if len(fit_by_state) < 2:
            continue

        # Now evaluate per state
        for st in list(fit_by_state.keys()):
            X_all, y_all, dates_all, feat_names, group_names = data_cache[st]
            dates_all = pd.to_datetime(dates_all)

            idx_train = (dates_all >= train_start) & (dates_all <= train_end)
            idx_val   = (dates_all >= val_start) & (dates_all <= train_end)
            idx_test  = (dates_all >= test_start) & (dates_all <= test_end)
            if idx_train.sum() < 10 or idx_val.sum() < 2 or idx_test.sum() < 2:
                continue

            X_tr, y_tr, d_tr = X_all[idx_train], y_all[idx_train], dates_all[idx_train]
            X_va, y_va = X_all[idx_val], y_all[idx_val]
            X_te, y_te, d_te = X_all[idx_test], y_all[idx_test], dates_all[idx_test]

            # ----- Baseline predictions -----
            ar_cols = np.arange(X_all.shape[1] - len(AR_LAGS), X_all.shape[1])
            Xtr_ar, Xva_ar, Xte_ar = X_tr[:, ar_cols], X_va[:, ar_cols], X_te[:, ar_cols]

            g = baseline_by_state[st]["gamma"]
            backcast_lag = (train_end - d_tr).days.astype(float).values
            w = np.exp(-g * backcast_lag)
            w = w / (w.max() if w.max() > 0 else 1.0)
            yhat_base = weighted_ols_fit_predict(Xtr_ar, y_tr, w, Xte_ar)

            # ----- SIS-GroupLasso target-only predictions -----
            target_feat_names = feat_names
            fit0 = fit_by_state[st]
            yhat_gl = predict_named(X_te, target_feat_names, fit0["coef"], fit0["intercept"])

            # ----- Aggregation -----
            # candidate states via sparsity index
            # all_states_here = [s for s in fit_by_state.keys() if s in d_by_state]
            all_states_here = [s for s in fit_by_state.keys() if s in d_by_state and s in VALID_STATES]
            cand_states = select_candidate_states(d_by_state, st, all_states_here, s=S_SCREEN, K=K_CANDIDATES)
            cand_all = [st] + cand_states

            # build Yhat on validation (on target state's X_va) and test (on target state's X_te)
            Yhat_val = []
            Yhat_test = []
            for s2 in cand_all:
                fit2 = fit_by_state[s2]
                Yhat_val.append(predict_named(X_va, target_feat_names, fit2["coef"], fit2["intercept"]))
                Yhat_test.append(predict_named(X_te, target_feat_names, fit2["coef"], fit2["intercept"]))
            Yhat_val = np.column_stack(Yhat_val)
            Yhat_test = np.column_stack(Yhat_test)

            w_agg = q_aggregate_weights(y_va, Yhat_val, total_step=10, selection=False, eps=1e-3)
            yhat_agg = Yhat_test @ w_agg
            for j, s2 in enumerate(cand_all):
                agg_rows.append({
                    "target_signal": target_name,
                    "target_state": st,
                    "train_end": train_end.date().isoformat(),
                    "candidate_state": s2,
                    "is_self": (j == 0),
                    "rank": j,                 # 0=self, 1..K are candidates
                    "weight": float(w_agg[j])  # weight for this candidate in Q-agg
                })
            for i, dt in enumerate(d_te):
                pred_rows.append({
                    "target_signal": target_name,
                    "geo_value": st,
                    "train_end": train_end.date().isoformat(),
                    "time_value": dt.date().isoformat(),
                    "model": "baseline_AR_decay",
                    "y_true": float(y_te[i]),
                    "y_pred": float(yhat_base[i]),
                })
                pred_rows.append({
                    "target_signal": target_name,
                    "geo_value": st,
                    "train_end": train_end.date().isoformat(),
                    "time_value": dt.date().isoformat(),
                    "model": "SIS_GroupLasso",
                    "y_true": float(y_te[i]),
                    "y_pred": float(yhat_gl[i]),
                })
                pred_rows.append({
                    "target_signal": target_name,
                    "geo_value": st,
                    "train_end": train_end.date().isoformat(),
                    "time_value": dt.date().isoformat(),
                    "model": "SIS_GroupLasso_Agg",
                    "y_true": float(y_te[i]),
                    "y_pred": float(yhat_agg[i]),
                })
    print(GL_ALPHA)
    pred_df = pd.DataFrame(pred_rows)
    agg_df = pd.DataFrame(agg_rows)
    agg_df["train_end"] = pd.to_datetime(agg_df["train_end"])
    # summary metrics
    summary = (
        pred_df.groupby(["target_signal", "model", "geo_value"])
        .apply(lambda g: pd.Series({
            "RMSE": np.sqrt(mean_squared_error(g["y_true"], g["y_pred"])),
            "MAE": mean_absolute_error(g["y_true"], g["y_pred"]),
            "N": len(g)
        }))
        .reset_index()
    )

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    pred_df["train_end"] = pd.to_datetime(pred_df["train_end"])
    pred_df["time_value"] = pd.to_datetime(pred_df["time_value"])

    def rmse(x):
        return np.sqrt(np.mean((x["y_true"].values - x["y_pred"].values) ** 2))

    # A) RMSE per retrain step (each step = next 30-day test period)
    rmse_by_step = (
        pred_df
        .groupby(["target_signal", "model", "train_end"])
        .apply(rmse)
        .rename("RMSE")
        .reset_index()
        .sort_values(["target_signal", "model", "train_end"])
    )

    sig = pred_df["target_signal"].iloc[0]
    tmp = rmse_by_step[rmse_by_step["target_signal"] == sig].copy()

    # Build a "target level" series aligned to train_end:
    # use mean(y_true) over the corresponding test block (all states pooled)
    target_level = (
        pred_df[pred_df["target_signal"] == sig]
        .groupby("train_end")["y_true"]
        .mean()
        .rename("target_mean")
        .reset_index()
    )

    # Merge so x-axis aligns
    tmp2 = tmp.merge(target_level, on="train_end", how="left")

    fig, ax = plt.subplots()

    # left axis: RMSE by model
    for m, g in tmp2.groupby("model"):
        ax.plot(g["train_end"], g["RMSE"], label=m)
    ax.set_xlabel("train_end (test is next 30d)")
    ax.set_ylabel("RMSE")

    # right axis: target signal level
    ax2 = ax.twinx()
    ax2.plot(target_level["train_end"], target_level["target_mean"], color = 'k', linestyle="--", label="Target mean (y_true)")
    ax2.set_ylabel("Target signal level (mean y_true)")

    # combined legend (both axes)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax.set_title(f"RMSE over time + target level: {sig}")
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    plt.show()


    print(pred_df.groupby(["target_signal", "model"]).apply(lambda g: pd.Series({
        "RMSE": np.sqrt(mean_squared_error(g["y_true"], g["y_pred"])),
        "MAE": mean_absolute_error(g["y_true"], g["y_pred"]),
        "N": len(g)
    })).reset_index())

    # exclude self
    cand_only = agg_df[~agg_df["is_self"]].copy()

    # frequency table
    freq = (
        cand_only
        .groupby(["target_signal", "target_state", "candidate_state"])
        .size()
        .rename("count")
        .reset_index()
    )

    from collections import defaultdict

    def most_frequent_candidates_dict(agg_df, top_k=5, drop_self=True, min_count=1):
        df = agg_df.copy()

        if drop_self and "is_self" in df.columns:
            df = df[~df["is_self"]]

        # frequency count
        freq = (
            df.groupby(["target_state", "candidate_state"])
            .size()
            .rename("count")
            .reset_index()
        )

        # filter low counts if wanted
        freq = freq[freq["count"] >= min_count]

        # build dict: state -> list of top_k candidates
        out = {}
        for st, g in freq.sort_values(["target_state", "count"], ascending=[True, False]).groupby("target_state"):
            out[st] = g["candidate_state"].head(top_k).tolist()

        return out

    # usage:
    top_candidates = most_frequent_candidates_dict(agg_df, top_k=5)
    top_candidates