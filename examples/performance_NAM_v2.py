import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

EVAL_BASE_DIR = "evaluate_modelperformance"
NAM_PLOT_DIR = os.path.join(EVAL_BASE_DIR, "NAM_v2")
os.makedirs(NAM_PLOT_DIR, exist_ok=True)


def _savefig(filename):
    path = os.path.join(NAM_PLOT_DIR, filename)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved → {path}")


def plot_actual_vs_predicted(df_pa, target):
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
    _savefig(f"actual_vs_pred_{target}.png")
    plt.close()


def plot_residuals(df_pa, target):
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
    _savefig(f"residuals_{target}.png")
    plt.close()


def plot_error_distribution(df_pa, target, percent=False):
    d = df_pa[df_pa["target"] == target]
    if len(d) == 0:
        print(f"No data for {target}")
        return
    plt.figure(figsize=(6, 4))
    values = d["percent_error"] if percent else d["abs_error"]
    sns.histplot(values, bins=30, kde=True)
    plt.xlabel("Percent Error (%)" if percent else "Absolute Error")
    plt.title(
        f"Error Distribution – {target} "
        + ("(%)" if percent else "(abs)")
    )
    plt.tight_layout()
    suffix = "pct" if percent else "abs"
    _savefig(f"error_dist_{target}_{suffix}.png")
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


def plot_error_bin_matrix(df_pa, targets, bins=(0, 5, 10, 20, 50, 100)):
    mat = build_error_bin_matrix(df_pa, targets, bins=bins)
    plt.figure(figsize=(1.4 * len(mat.columns), 0.5 * len(mat.index) + 3))
    sns.heatmap(mat, annot=True, fmt="d", cmap="viridis", cbar_kws={"label": "Count"})
    plt.xlabel("Percent Error Bin")
    plt.ylabel("Target")
    plt.title("Error-bin Confusion Matrix (NAM)")
    plt.tight_layout()
    _savefig("error_bin_confusion_matrix.png")
    plt.close()


def generate_all_plots_from_csv():
    csv_path = os.path.join(NAM_PLOT_DIR, "predicted_vs_actual_NAM_test.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find CSV at {csv_path}. "
            f"Run nam_v2_train.py first to generate it."
        )

    df_pa = pd.read_csv(csv_path)
    targets = sorted(df_pa["target"].unique())

    print("\n=== Generating per-target NAM plots from CSV ===")
    for t in targets:
        d = df_pa[df_pa["target"] == t]
        if len(d) == 0:
            print(f"[{t}] – no data, skipping.")
            continue
        print(f"[{t}] – plotting ...")
        plot_actual_vs_predicted(df_pa, t)
        plot_residuals(df_pa, t)
        plot_error_distribution(df_pa, t, percent=False)
        plot_error_distribution(df_pa, t, percent=True)

    print("\n=== Generating global NAM error-bin matrix ===")
    plot_error_bin_matrix(df_pa, targets, bins=(0, 5, 10, 20, 50, 100))
    print(f"\nAll NAM plots saved in '{NAM_PLOT_DIR}'.")


if __name__ == "__main__":
    generate_all_plots_from_csv()
