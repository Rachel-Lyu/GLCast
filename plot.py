import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


BASE_MODEL = "baseline_AR_decay"
GL_MODEL = "SIS_GroupLasso"
AGG_MODEL = "SIS_GroupLasso_Agg"
REQUIRED_MODELS = [BASE_MODEL, GL_MODEL, AGG_MODEL]
BASE_DIFF_TO_COLOR = {
    "baseline - GL": "tab:blue",
    "baseline - Agg": "tab:green",
    "GL - Agg": "tab:orange",
}



def _safe_slug(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "plot"



def _rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))



def _discover_run_dirs(root_dir: str, dataset: Optional[str], alpha: Optional[str]) -> List[Path]:
    if not dataset:
        raise ValueError("--dataset is required when --run-dir is omitted.")

    dataset_dir = Path(root_dir).expanduser() / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset output directory not found: {dataset_dir}")

    if alpha is not None:
        alpha_dir = dataset_dir / f"alpha_{alpha}"
        if not alpha_dir.exists():
            raise FileNotFoundError(f"Requested alpha directory not found: {alpha_dir}")
        candidates = [alpha_dir]
    else:
        candidates = sorted(
            p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("alpha_")
        )

    run_dirs = [p for p in candidates if (p / "predictions.csv").exists() and (p / "rmse_by_step.csv").exists()]
    if not run_dirs:
        raise FileNotFoundError(
            f"No runnable output folders were found under {dataset_dir}. "
            "Expected predictions.csv and rmse_by_step.csv."
        )
    return run_dirs



def _load_run_tables(run_dir: Path):
    pred_path = run_dir / "predictions.csv"
    rmse_path = run_dir / "rmse_by_step.csv"
    metadata_path = run_dir / "run_metadata.json"

    pred_df = pd.read_csv(pred_path)
    rmse_df = pd.read_csv(rmse_path)
    meta = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    if pred_df.empty:
        raise ValueError(f"predictions.csv is empty in {run_dir}")

    pred_df["train_end"] = pd.to_datetime(pred_df["train_end"])
    pred_df["time_value"] = pd.to_datetime(pred_df["time_value"])
    if not rmse_df.empty:
        rmse_df["train_end"] = pd.to_datetime(rmse_df["train_end"])

    return pred_df, rmse_df, meta



def _infer_alpha_label(run_dir: Path, pred_df: pd.DataFrame, meta: dict) -> str:
    if "alpha" in meta:
        return str(meta["alpha"])
    if "alpha" in pred_df.columns and pred_df["alpha"].notna().any():
        vals = pred_df["alpha"].dropna().unique().tolist()
        if len(vals) == 1:
            return str(vals[0])
    return run_dir.name.replace("alpha_", "")



def _color_boxes_by_diff(bp, columns: List[str]):
    for i, box_artist in enumerate(bp["boxes"]):
        colname = columns[i]
        color = BASE_DIFF_TO_COLOR.get(colname, "lightgray")
        box_artist.set_facecolor(color)
        box_artist.set_edgecolor("black")
        box_artist.set_alpha(0.8)

    for w in bp["whiskers"]:
        w.set(color="black", linewidth=1)
    for c in bp["caps"]:
        c.set(color="black", linewidth=1)
    for m in bp["medians"]:
        m.set(color="darkred", linewidth=2)



def make_boxplot_rmse_differences(
    pred_df: pd.DataFrame,
    run_dir: Path,
    plots_dir: Path,
    target_signal: str,
    alpha_label: str,
    show: bool = False,
    dpi: int = 180,
):
    df_sig = pred_df[pred_df["target_signal"] == target_signal].copy()
    if df_sig.empty:
        return None

    rows = []
    for (geo_value, model), g in df_sig.groupby(["geo_value", "model"]):
        rows.append({
            "geo_value": geo_value,
            "model": model,
            "RMSE": _rmse(g["y_true"], g["y_pred"]),
        })
    rmse_state = pd.DataFrame(rows)
    if rmse_state.empty:
        return None

    rmse_wide = rmse_state.pivot(index="geo_value", columns="model", values="RMSE")
    missing = [m for m in REQUIRED_MODELS if m not in rmse_wide.columns]
    if missing:
        print(f"[skip] {run_dir}: missing models for boxplot of {target_signal}: {missing}")
        return None

    rmse_wide = rmse_wide[REQUIRED_MODELS].dropna()
    if rmse_wide.empty:
        print(f"[skip] {run_dir}: no states with all required models for {target_signal}")
        return None

    diff_df = pd.DataFrame(index=rmse_wide.index)
    diff_df["baseline - GL"] = rmse_wide[BASE_MODEL] - rmse_wide[GL_MODEL]
    diff_df["baseline - Agg"] = rmse_wide[BASE_MODEL] - rmse_wide[AGG_MODEL]
    diff_df["GL - Agg"] = rmse_wide[GL_MODEL] - rmse_wide[AGG_MODEL]

    cols = diff_df.columns.tolist()
    fig, ax = plt.subplots(figsize=(9, 4))
    bp = ax.boxplot(
        [diff_df[c].dropna().values for c in cols],
        labels=cols,
        patch_artist=True,
        showfliers=False,
    )
    _color_boxes_by_diff(bp, cols)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("RMSE difference (positive = second model better)")
    ax.set_title(f"RMSE difference across states: {target_signal} | alpha={alpha_label}")
    ax.grid(True, axis="y", alpha=0.2)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    out_path = plots_dir / f"boxplot_rmse_differences_{_safe_slug(target_signal)}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path



def make_rmse_over_time_plot(
    pred_df: pd.DataFrame,
    rmse_by_step_df: pd.DataFrame,
    run_dir: Path,
    plots_dir: Path,
    target_signal: str,
    alpha_label: str,
    show: bool = False,
    dpi: int = 180,
):
    pred_sig = pred_df[pred_df["target_signal"] == target_signal].copy()
    tmp2 = rmse_by_step_df[rmse_by_step_df["target_signal"] == target_signal].copy()

    if pred_sig.empty or tmp2.empty:
        return None

    target_level = (
        pred_sig[["train_end", "geo_value", "time_value", "y_true"]]
        .drop_duplicates()
        .groupby("train_end", as_index=False)
        .agg(target_mean=("y_true", "mean"))
        .sort_values("train_end")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name, g in tmp2.groupby("model"):
        g = g.sort_values("train_end")
        ax.plot(g["train_end"].to_numpy(), g["RMSE"].to_numpy(), marker="o", label=model_name)
    ax.set_xlabel("train_end (test is next 30d)")
    ax.set_ylabel("RMSE")

    ax2 = ax.twinx()
    ax2.plot(
        target_level["train_end"].to_numpy(),
        target_level["target_mean"].to_numpy(),
        color="k",
        linestyle="--",
        label="Target mean (y_true)",
    )
    ax2.set_ylabel("Target signal level (mean y_true)")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax.set_title(f"RMSE over time + target level: {target_signal} | alpha={alpha_label}")
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()

    out_path = plots_dir / f"rmse_over_time_target_level_{_safe_slug(target_signal)}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path



def create_plots_for_run(run_dir: Path, plots_subdir: str = "plots", show: bool = False, dpi: int = 180):
    pred_df, rmse_by_step_df, meta = _load_run_tables(run_dir)
    plots_dir = run_dir / plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)

    alpha_label = _infer_alpha_label(run_dir, pred_df, meta)
    target_signals = sorted(pred_df["target_signal"].dropna().unique().tolist())

    saved_paths = []
    for sig in target_signals:
        p1 = make_boxplot_rmse_differences(
            pred_df=pred_df,
            run_dir=run_dir,
            plots_dir=plots_dir,
            target_signal=sig,
            alpha_label=alpha_label,
            show=show,
            dpi=dpi,
        )
        if p1 is not None:
            saved_paths.append(p1)

        p2 = make_rmse_over_time_plot(
            pred_df=pred_df,
            rmse_by_step_df=rmse_by_step_df,
            run_dir=run_dir,
            plots_dir=plots_dir,
            target_signal=sig,
            alpha_label=alpha_label,
            show=show,
            dpi=dpi,
        )
        if p2 is not None:
            saved_paths.append(p2)

    return saved_paths