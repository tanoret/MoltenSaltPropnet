# evaluate_catboostv2.py
"""The script with the undeep catboost version evaluation plots."""
# evaluate_catboostv2.py
#
# Plotting & automatic diagnostics for catboostv2
# Reads artifacts from train_catboostv2.py and saves plots
# into evaluate_modelperformance/catboost_v1

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

import shap  # for optional SHAP plots

# ---------------------------------
# PATHS
# ---------------------------------

CSV_PATH_RAW = "/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv"

EVAL_ROOT = "evaluate_modelperformance"
EVAL_CATBOOST_V1_DIR = os.path.join(EVAL_ROOT, "catboost_v1")
os.makedirs(EVAL_CATBOOST_V1_DIR, exist_ok=True)

PRED_VS_ACTUAL_PATH = os.path.join(EVAL_CATBOOST_V1_DIR, "predicted_vs_actual_catboostv2.csv")
CATBOOST_EVALS_PATH = os.path.join(EVAL_CATBOOST_V1_DIR, "catboostv2_evals_results.json")

def _savefig(filename):
    full_path = os.path.join(EVAL_CATBOOST_V1_DIR, filename)
    plt.savefig(full_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot → {full_path}")



# PLOTTING FUNCTIONS


def plot_actual_vs_predicted(df_pa, target, save=True):
    d = df_pa[df_pa["target"] == target]
    if len(d) == 0:
        print(f"No data for {target}")
        return

    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=d, x="actual", y="predicted", alpha=0.6)
    lo = min(d["actual"].min(), d["predicted"].min())
    hi = max(d["actual"].max(), d["predicted"].max())
    plt.plot([lo, hi], [lo, hi], "r--")

    plt.title(f"Actual vs Predicted – {target} (catboostv2)")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()

    if save:
        _savefig(f"catboostv2_actual_vs_pred_{target}.png")

    plt.close()


def plot_residuals(df_pa, target, save=True):
    d = df_pa[df_pa["target"] == target].copy()
    if len(d) == 0:
        print(f"No data for {target}")
        return

    d["residual"] = d["predicted"] - d["actual"]

    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=d, x="actual", y="residual", alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")

    plt.title(f"Residuals – {target} (catboostv2)")
    plt.xlabel("Actual")
    plt.ylabel("Residual")
    plt.tight_layout()

    if save:
        _savefig(f"catboostv2_residuals_{target}.png")

    plt.close()


def plot_error_distribution(df_pa, target, percent=False, save=True):
    d = df_pa[df_pa["target"] == target]
    if len(d) == 0:
        print(f"No data for {target}")
        return

    plt.figure(figsize=(6, 4))
    values = d["percent_error"] if percent else d["abs_error"]
    sns.histplot(values, bins=30, kde=True)

    plt.xlabel("Percent Error (%)" if percent else "Absolute Error")
    plt.title(f"Error Distribution – {target} (catboostv2)")
    plt.tight_layout()

    if save:
        suffix = "pct" if percent else "abs"
        _savefig(f"catboostv2_error_dist_{target}_{suffix}.png")

    plt.close()


def plot_all_targets_grid(df_pa, targets, cols=4, save=True):
    n = len(targets)
    if n == 0:
        return

    rows_grid = (n + cols - 1) // cols
    plt.figure(figsize=(cols * 4, rows_grid * 3.2))

    for i, t in enumerate(targets):
        d = df_pa[df_pa["target"] == t]
        if len(d) == 0:
            continue

        plt.subplot(rows_grid, cols, i + 1)
        sns.scatterplot(data=d, x="actual", y="predicted", alpha=0.6)
        lo = min(d["actual"].min(), d["predicted"].min())
        hi = max(d["actual"].max(), d["predicted"].max())
        plt.plot([lo, hi], [lo, hi], "r--")
        plt.title(t)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")

    plt.tight_layout()
    if save:
        _savefig("catboostv2_all_targets_grid.png")

    plt.close()


def plot_confusion_style_heatmap(df_pa, target, bins=40, save=True):
    d = df_pa[df_pa["target"] == target]
    if len(d) == 0:
        print(f"No data for {target}")
        return

    actual = d["actual"].values
    pred   = d["predicted"].values

    vmin = min(actual.min(), pred.min())
    vmax = max(actual.max(), pred.max())

    plt.figure(figsize=(6, 5))
    plt.hist2d(actual, pred, bins=bins,
               range=[[vmin, vmax], [vmin, vmax]],
               cmap="viridis")
    plt.plot([vmin, vmax], [vmin, vmax], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Confusion-style Heatmap – {target} (catboostv2)")
    plt.colorbar(label="Count")
    plt.tight_layout()

    if save:
        _savefig(f"catboostv2_confusion_heatmap_{target}.png")

    plt.close()


# --------- SHAP & FEATURE IMPORTANCE (optional) ---------

def plot_feature_importance_catboost(model, feature_names, target, save=True):
    """Plot CatBoost feature importance (requires trained CatBoost model)."""
    importances = model.feature_importances_

    plt.figure(figsize=(8, 6))
    idx = np.argsort(importances)[::-1]
    plt.barh(np.array(feature_names)[idx][::-1], importances[idx][::-1])
    plt.title(f"Feature Importance – {target} (catboostv2)")
    plt.xlabel("Importance")
    plt.tight_layout()

    if save:
        _savefig(f"catboostv2_feature_importance_{target}.png")

    plt.close()


def plot_shap_summary_catboost(model, X_sample, target, save=True):
    """Plot SHAP summary (global importance + distribution) for a CatBoost model."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
        plot_size=(8, 6),
        max_display=20
    )

    if save:
        _savefig(f"catboostv2_shap_summary_{target}.png")

    plt.close()


def plot_shap_dependence_catboost(model, X_sample, feature_name, target, save=True):
    """SHAP dependence for a single feature (requires CatBoost model + X_sample)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.dependence_plot(
        feature_name,
        shap_values,
        X_sample,
        show=False
    )

    if save:
        _savefig(f"catboostv2_shap_dependence_{target}_{feature_name}.png")

    plt.close()


# --------- ERROR BIN MATRIX & ERROR BY SYSTEM ---------

def build_error_bin_matrix(df_pa, targets, bins=(0, 5, 10, 20, 50, 100)):
    labels = []
    for i in range(len(bins) - 1):
        labels.append(f"{bins[i]}–{bins[i+1]}%")
    labels.append(f">{bins[-1]}%")

    mat = pd.DataFrame(0, index=targets, columns=labels)

    for t in targets:
        d = df_pa[df_pa["target"] == t]
        if len(d) == 0:
            continue
        errs = d["percent_error"].values
        idxs = np.digitize(errs, bins)

        for idx in idxs:
            if idx < len(bins):
                col = labels[idx-1] if idx > 0 else labels[0]
            else:
                col = labels[-1]
            mat.loc[t, col] += 1

    return mat


def plot_error_bin_matrix(df_pa, targets, bins=(0, 5, 10, 20, 50, 100), save=True):
    mat = build_error_bin_matrix(df_pa, targets, bins=bins)

    plt.figure(figsize=(1.4 * len(mat.columns), 0.5 * len(mat.index) + 3))
    sns.heatmap(mat, annot=True, fmt="d", cmap="viridis", cbar_kws={"label": "Count"})
    plt.xlabel("Percent Error Bin")
    plt.ylabel("Target")
    plt.title("Error-bin Confusion Matrix (catboostv2)")
    plt.tight_layout()

    if save:
        _savefig("catboostv2_error_bin_confusion_matrix.png")

    plt.close()


def plot_error_by_system(df_raw, df_pa, target, top_n_systems=10, percent=True, save=True):
    d = df_pa[df_pa["target"] == target]
    if len(d) == 0:
        print(f"No data for {target}")
        return

    if "System" not in df_raw.columns:
        print("System column not in df_raw; cannot plot error by system.")
        return

    merged = d.merge(df_raw[["System"]], left_on="index", right_index=True, how="left")

    err_col = "percent_error" if percent else "abs_error"
    ylabel  = "Percent Error (%)" if percent else "Absolute Error"

    top = merged["System"].value_counts().head(top_n_systems).index
    merged = merged[merged["System"].isin(top)]

    if len(merged) == 0:
        print(f"No data for {target} after filtering by top systems.")
        return

    plt.figure(figsize=(max(8, 0.7 * len(top)), 4))
    sns.boxplot(data=merged, x="System", y=err_col)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Error by System – {target} (catboostv2)")
    plt.ylabel(ylabel)
    plt.tight_layout()

    if save:
        _savefig(f"catboostv2_error_by_system_{target}.png")

    plt.close()


def plot_catboost_learning_curve(target, evals_results, save=True):
    if target not in evals_results:
        print(f"No CatBoost evals found for {target}.")
        return

    ev = evals_results[target]
    train_loss = ev["learn"]["RMSE"]
    val_loss = ev["validation"]["RMSE"]

    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, label="Train RMSE")
    plt.plot(val_loss, label="Validation RMSE")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title(f"CatBoost Learning Curve – {target} (catboostv2)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save:
        _savefig(f"catboostv2_learning_curve_{target}.png")

    plt.close()


def generate_all_plots(df_raw, pred_df, targets, evals_results, top_n_systems=10):
    """
    Generate all diagnostic plots for all targets and save them
    into evaluate_modelperformance/catboost_v1.
    """
    print("\n=== catboostv2: Generating per-target plots ===")
    for t in targets:
        d = pred_df[pred_df["target"] == t]
        if len(d) == 0:
            print(f"[{t}] – no data, skipping.")
            continue

        print(f"[{t}] – plotting...")

        plot_actual_vs_predicted(pred_df, t, save=True)
        plot_residuals(pred_df, t, save=True)
        plot_error_distribution(pred_df, t, percent=False, save=True)
        plot_error_distribution(pred_df, t, percent=True,  save=True)
        plot_confusion_style_heatmap(pred_df, t, bins=40, save=True)
        plot_error_by_system(df_raw, pred_df, t,
                             top_n_systems=top_n_systems,
                             percent=True,
                             save=True)

        # Use presence in evals_results as proxy for CatBoost models
        if t in evals_results:
            plot_catboost_learning_curve(t, evals_results, save=True)

    print("\n=== catboostv2: Generating global plots (all targets) ===")
    plot_all_targets_grid(pred_df, targets, cols=4, save=True)
    plot_error_bin_matrix(pred_df, targets, bins=(0, 5, 10, 20, 50, 100), save=True)
    print("\nAll catboostv2 plots generated and saved in 'evaluate_modelperformance/catboost_v1'.")


# ---------------------------------
# MAIN
# ---------------------------------

if __name__ == "__main__":
    print("\n=== catboostv2: Loading data for evaluation plots ===")
    df_raw = pd.read_csv(CSV_PATH_RAW)
    pred_vs_actual = pd.read_csv(PRED_VS_ACTUAL_PATH)

    with open(CATBOOST_EVALS_PATH, "r") as f:
        evals_results = json.load(f)

    targets = sorted(pred_vs_actual["target"].unique())
    print("Targets found in predicted_vs_actual_catboostv2.csv:", targets)

    generate_all_plots(df_raw, pred_vs_actual, targets, evals_results, top_n_systems=10)
    print("\nDone with catboostv2 evaluation.")
