
"""
Cross-validate ResNetMetaTrainerKNN on MSTDB data, evaluate, visualize,
and store outputs.

All outputs are saved under:
    evaluate_modelperformance/resnet_KNN/
"""

import os
import sys
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processing_mstdb.resnet_KNN import (
    ResNetMetaTrainerKNN,
    TARGETS,
    DERIVED_PROPS,
)

OUTDIR = os.path.join("evaluate_modelperformance", "resnet_KNN")
os.makedirs(OUTDIR, exist_ok=True)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _rel_mse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative MSE as percentage of <y^2>."""
    mse = mean_squared_error(y_true, y_pred)
    denom = np.mean(y_true ** 2)
    if not np.isfinite(denom) or denom == 0.0:
        denom = 1e-12
    return 100.0 * mse / denom


def save_plot(filename: str):
    plt.tight_layout()
    path = os.path.join(OUTDIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


# -------------------------------------------------------------------
# K-fold CV wrapper
# -------------------------------------------------------------------

def cross_validate_resnet_KNN(
    df: pd.DataFrame,
    targets: List[str],
    derived_props: List,
    k: int = 3,
) -> (List[Dict], Dict[str, Dict[str, np.ndarray]]):
    """
    Perform K-fold CV on the ResNetMetaTrainerKNN.

    For each fold:
      - Train on (k-1)/k of data
      - Evaluate on held-out 1/k set
      - Return per-target metrics on that held-out set.

    Returns
    -------
    cv_results : list of dict (one per fold)
       Each dict has key "test" with "per_target" metrics.
    pooled_scatter : dict
       pooled_scatter[target]["y_true"], pooled_scatter[target]["y_pred"]
       for making global true-vs-pred plots.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    indices = np.arange(len(df))

    cv_results: List[Dict] = []
    pooled_scatter: Dict[str, Dict[str, List[float]]] = {
        t: {"y_true": [], "y_pred": []} for t in targets
    }

    fold_id = 0
    for train_idx, test_idx in kf.split(indices):
        fold_id += 1
        print(f"\n================ FOLD {fold_id}/{k} ================")
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test  = df.iloc[test_idx].reset_index(drop=True)

        # ---- Train trainer on TRAIN-portion only --------------------
        trainer = ResNetMetaTrainerKNN(
            df_train,
            targets,
            derived_props,
        )
        print(
            f"Using {len(trainer.present_targets)} properties in this fold: "
            + ", ".join(trainer.present_targets)
        )

        trainer.train_base()
        trainer.train_meta()

        # optional: internal validation evaluation (on trainer.va_idx)
        _ = trainer.evaluate(return_dict=False)

        # ---- Evaluate on held-out TEST portion ----------------------
        per_target_metrics = {}
        rel_mses, r2s = [], []

        # accumulate predictions per target for this fold
        fold_true: Dict[str, List[float]] = {t: [] for t in trainer.present_targets}
        fold_pred: Dict[str, List[float]] = {t: [] for t in trainer.present_targets}

        for _, row in df_test.iterrows():
            # ground truth for all targets (some may be NaN)
            # composition from the row
            try:
                comp = trainer.row_composition(row)
            except Exception as e:
                print(f"Skipping row due to composition parse error: {e}")
                continue

            try:
                preds = trainer.predict(comp)
            except Exception as e:
                print(f"Skipping row due to prediction error: {e}")
                continue

            for prop in trainer.present_targets:
                y_val = row.get(prop, np.nan)
                if pd.notna(y_val) and np.isfinite(y_val):
                    fold_true[prop].append(float(y_val))
                    fold_pred[prop].append(float(preds[prop]))

        print(f"\nFold {fold_id} — held-out test metrics:")
        for prop in trainer.present_targets:
            yt = np.array(fold_true[prop], dtype=float)
            yp = np.array(fold_pred[prop], dtype=float)
            n = len(yt)
            if n < 2:
                m_rel = float("nan")
                r2 = float("nan")
            else:
                m_rel = _rel_mse_pct(yt, yp)
                r2 = r2_score(yt, yp)
            per_target_metrics[prop] = {
                "relMSE_pct": float(m_rel),
                "R2": float(r2),
                "n": int(n),
            }

            print(
                f" • {prop:<8s}: {m_rel:8.2f}%   R²={r2:+.3f}   (n={n})"
            )

            rel_mses.append(m_rel)
            r2s.append(r2)

            # Add to pooled scatter
            pooled_scatter[prop]["y_true"].extend(yt.tolist())
            pooled_scatter[prop]["y_pred"].extend(yp.tolist())

        avg_rel = float(np.nanmean(rel_mses)) if rel_mses else float("nan")
        avg_r2  = float(np.nanmean(r2s)) if r2s else float("nan")

        cv_results.append(
            {
                "fold": fold_id,
                "test": {
                    "avg_relMSE_pct": avg_rel,
                    "avg_R2": avg_r2,
                    "per_target": per_target_metrics,
                },
            }
        )

    # convert pooled scatter lists to numpy arrays
    pooled_np = {}
    for t in targets:
        yt = np.array(pooled_scatter[t]["y_true"], dtype=float)
        yp = np.array(pooled_scatter[t]["y_pred"], dtype=float)
        pooled_np[t] = {"y_true": yt, "y_pred": yp}

    return cv_results, pooled_np


# -------------------------------------------------------------------
# Plotting functions
# -------------------------------------------------------------------

def plot_cv_r2_boxplot(cv_results: List[Dict]):
    """Boxplot of R² across folds per target."""

    r2_per_target: Dict[str, List[float]] = {}

    for fold_res in cv_results:
        test = fold_res.get("test", {})
        pt = test.get("per_target", {})
        for t, m in pt.items():
            r2 = m.get("R2", float("nan"))
            r2_per_target.setdefault(t, []).append(r2)

    if not r2_per_target:
        print("cv_results has no per-target R²; skipping CV boxplot.")
        return

    targets = sorted(r2_per_target.keys())
    data = [r2_per_target[t] for t in targets]

    plt.figure(figsize=(12, 5))
    plt.boxplot(data, labels=targets, showmeans=True)
    plt.axhline(0, color="black")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("R² (test across folds)")
    plt.title("ResNet_KNN — cross-validation R² stability per target")
    save_plot("cv_r2_boxplot.png")


def plot_mean_r2_with_ci(cv_results: List[Dict]):
    """Mean R² ± 95% CI across folds per target."""

    r2_per_target: Dict[str, List[float]] = {}

    for fold_res in cv_results:
        test = fold_res.get("test", {})
        pt = test.get("per_target", {})
        for t, m in pt.items():
            r2 = m.get("R2", float("nan"))
            r2_per_target.setdefault(t, []).append(r2)

    if not r2_per_target:
        print("No R² values to plot mean R²; skipping.")
        return

    targets = sorted(r2_per_target.keys())
    means, cis = [], []

    for t in targets:
        vals = np.array(r2_per_target[t], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            means.append(np.nan)
            cis.append(0.0)
            continue
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci = 1.96 * s / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        means.append(m)
        cis.append(ci)

    x = np.arange(len(targets))
    plt.figure(figsize=(12, 5))
    plt.bar(x, means, yerr=cis, capsize=5)
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(x, targets, rotation=45, ha="right")
    plt.ylabel("Mean R² (3-fold CV)")
    plt.title("ResNet_KNN — Mean R² per target (with 95% CI)")
    save_plot("mean_r2_with_ci.png")


def plot_true_vs_pred_scatter(pooled_scatter: Dict[str, Dict[str, np.ndarray]], min_n: int = 5):
    """
    Scatter plots of true vs predicted for each target,
    pooled over all CV folds.
    """
    targets = sorted(pooled_scatter.keys())
    n_targets = len(targets)
    n_cols = 3
    n_rows = int(np.ceil(n_targets / n_cols))

    plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    plot_idx = 1

    for t in targets:
        yt = pooled_scatter[t]["y_true"]
        yp = pooled_scatter[t]["y_pred"]
        mask = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[mask]
        yp = yp[mask]
        if len(yt) < min_n:
            continue

        plt.subplot(n_rows, n_cols, plot_idx)
        plt.scatter(yt, yp, alpha=0.5, s=10)
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        plt.title(t)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plot_idx += 1

    plt.suptitle("ResNet_KNN — True vs predicted (pooled over CV folds)", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_plot("true_vs_pred_all_cv.png")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    # 1. Load data
    csv_path = "/Users/meggie/Documents/MoltenSaltPropnet/data/mstdb_processed.csv"
    df = pd.read_csv(csv_path).rename(columns=str.strip)

    # 2. K-fold CV
    K_FOLDS = 3
    cv_results, pooled_scatter = cross_validate_resnet_KNN(
        df,
        TARGETS,
        DERIVED_PROPS,
        k=K_FOLDS,
    )

    # 3. Save raw CV results
    with open(os.path.join(OUTDIR, "cv_results_resnet_KNN.json"), "w") as f:
        json.dump(cv_results, f, indent=4)
    print(f"\nCV results JSON saved under: {OUTDIR}")

    # 4. Plots
    plot_cv_r2_boxplot(cv_results)
    plot_mean_r2_with_ci(cv_results)
    plot_true_vs_pred_scatter(pooled_scatter)

    print("\nAll plots saved under:", OUTDIR)


if __name__ == "__main__":
    main()
