"""The Idea was to use something more narrow since there is missing data and some values hard to evaluate
the ide was to make kind of base and meta layer like the complex ones of you that you made
as well since catbbooost is just working with a lot of values thats why its just for some catboost and than later something else.

"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


CSV_PATH_RAW = "/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv"
PRED_VS_ACTUAL_PATH = "predicted_vs_actual_meta.csv"
CATBOOST_EVALS_PATH = "catboost_evals_results.json"
BASE_PLOT_DIR = "evaluate_modelperformance"
EVAL_PLOT_DIR = os.path.join(BASE_PLOT_DIR, "catboost_meta")
os.makedirs(EVAL_PLOT_DIR, exist_ok=True)


def _savefig(filename):
    full_path = os.path.join(EVAL_PLOT_DIR, filename)
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

    plt.title(f"Actual vs Predicted – {target}")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()

    if save:
        _savefig(f"actual_vs_pred_{target}.png")

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

    plt.title(f"Residuals – {target}")
    plt.xlabel("Actual")
    plt.ylabel("Residual")
    plt.tight_layout()

    if save:
        _savefig(f"residuals_{target}.png")

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
    plt.title(f"Error Distribution – {target}")
    plt.tight_layout()

    if save:
        suffix = "pct" if percent else "abs"
        _savefig(f"error_dist_{target}_{suffix}.png")

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
    plt.title(f"Confusion-style Heatmap – {target}")
    plt.colorbar(label="Count")
    plt.tight_layout()

    if save:
        _savefig(f"confusion_heatmap_{target}.png")

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
        _savefig("all_targets_grid.png")

    plt.close()


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
                col = labels[idx - 1] if idx > 0 else labels[0]
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
    plt.title("Error-bin Confusion Matrix")
    plt.tight_layout()

    if save:
        _savefig("error_bin_confusion_matrix.png")

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
    plt.title(f"Error by System – {target}")
    plt.ylabel(ylabel)
    plt.tight_layout()

    if save:
        _savefig(f"error_by_system_{target}.png")

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
    plt.title(f"CatBoost Learning Curve – {target}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save:
        _savefig(f"learning_curve_{target}.png")

    plt.close()


def generate_all_plots(df_raw, pred_df, targets, evals_results, top_n_systems=10):
    """
    Generate all diagnostic plots for all targets and save them
    into evaluate_modelperformance/catboost_meta.
    """
    print("\n=== Generating per-target plots ===")
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

        # CatBoost learning curves
        if t in evals_results:
            plot_catboost_learning_curve(t, evals_results, save=True)

    print("\n=== Generating global plots (all targets) ===")
    plot_all_targets_grid(pred_df, targets, cols=4, save=True)
    plot_error_bin_matrix(pred_df, targets, bins=(0, 5, 10, 20, 50, 100), save=True)
    print("\nAll plots generated and saved in 'evaluate_modelperformance/catboost_meta'.")



# MAIN


if __name__ == "__main__":
    print("\n=== Loading data for evaluation plots ===")
    df_raw = pd.read_csv(CSV_PATH_RAW)
    pred_vs_actual = pd.read_csv(PRED_VS_ACTUAL_PATH)

    with open(CATBOOST_EVALS_PATH, "r") as f:
        evals_results = json.load(f)

    targets = sorted(pred_vs_actual["target"].unique())
    print("Targets found in predicted_vs_actual_meta.csv:", targets)

    generate_all_plots(df_raw, pred_vs_actual, targets, evals_results, top_n_systems=10)
    print("\nDone.")
