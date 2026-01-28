import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Local import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processing_saltdblean.mlp_trainer import (
    MLPMetaTrainer,
    TARGETS,
    DERIVED_PROPS,
    ELEMENT_FEATURE_COLS,
)

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz_with_ionic_polarizability.csv"

BASE_OUTDIR = os.path.join("data", "mlp_compare")
PLOT_DIR = os.path.join(BASE_OUTDIR, "plots")
MODELS_A_DIR = os.path.join(BASE_OUTDIR, "models_A_without_elements")
MODELS_B_DIR = os.path.join(BASE_OUTDIR, "models_B_with_elements")

TEMPERATURE = 900
MAX_TARGETS_PER_FIG = 8
ELEMENT_FILTERS = ["Cl", "F"]

SINGLE_COMPOSITIONS = {
    "NaCl (50-50 atoms)": {"Na": 0.5, "Cl": 0.5},
}

EMBEDDING_METHOD = "pca"
N_COMPONENTS = 32

SEED = 42
np.random.seed(SEED)

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODELS_A_DIR, exist_ok=True)
os.makedirs(MODELS_B_DIR, exist_ok=True)


# ============================================================
# Load data
# ============================================================
import pandas as pd
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# ============================================================
# Shared splits for A and B
# ============================================================
idx_all = np.arange(len(df))
tr_idx, te_idx = train_test_split(idx_all, test_size=0.20, random_state=SEED)
tr_idx, va_idx = train_test_split(tr_idx, test_size=0.20, random_state=SEED)
splits = (tr_idx, va_idx, te_idx)

# ============================================================
# Build trainers A and B
# ============================================================
trainerA = MLPMetaTrainer(
    df=df,
    target_cols=TARGETS,
    derived_props=DERIVED_PROPS,
    element_feature_cols=ELEMENT_FEATURE_COLS,
    use_element_features=False,
    embedding_method=EMBEDDING_METHOD,
    n_components=N_COMPONENTS,
    splits=splits,
    model_dir=MODELS_A_DIR,
)

trainerB = MLPMetaTrainer(
    df=df,
    target_cols=TARGETS,
    derived_props=DERIVED_PROPS,
    element_feature_cols=ELEMENT_FEATURE_COLS,
    use_element_features=True,
    embedding_method=EMBEDDING_METHOD,
    n_components=N_COMPONENTS,
    splits=splits,
    model_dir=MODELS_B_DIR,
)

assert trainerA.present_targets == trainerB.present_targets, "A/B targets mismatch!"

# ============================================================
# SNIPPET C — Helper to format dict → readable text
# ============================================================
def format_metrics_txt(result_dict):
    lines = []
    lines.append(f"AVG MSE%: {result_dict['avg_mse_pct']:.4f}")
    lines.append(f"AVG R2:   {result_dict['avg_r2']:.4f}")
    lines.append("")
    lines.append("PER-TARGET RESULTS:")
    for tgt, vals in result_dict["per_target"].items():
        lines.append(
            f"  {tgt:10s} | MSE%={vals['MSE_pct']:.4f} | R2={vals['R2']:.4f}"
        )
    return "\n".join(lines)


# ============================================================
# TRAIN A
# ============================================================
print("\n==============================")
print("TRAIN A (MLP): WITHOUT element features")
print("==============================")
trainerA.train_base()
trainerA.train_meta()
resA_val = trainerA.evaluate("val", return_dict=True)
resA_test = trainerA.evaluate("test", return_dict=True)

# ============================================================
# SNIPPET A — save metrics for A
# ============================================================
with open(os.path.join(BASE_OUTDIR, "metrics_A.txt"), "w") as f:
    f.write("====== MODEL A (NO ELEMENT FEATURES) ======\n\n")
    f.write("VALIDATION:\n")
    f.write(format_metrics_txt(resA_val))
    f.write("\n\nTEST:\n")
    f.write(format_metrics_txt(resA_test))
print("Saved metrics_A.txt")


# ============================================================
# TRAIN B
# ============================================================
print("\n==============================")
print("TRAIN B (MLP): WITH element features")
print("==============================")
trainerB.train_base()
trainerB.train_meta()
resB_val = trainerB.evaluate("val", return_dict=True)
resB_test = trainerB.evaluate("test", return_dict=True)

# ============================================================
# SNIPPET B — save metrics for B
# ============================================================
with open(os.path.join(BASE_OUTDIR, "metrics_B.txt"), "w") as f:
    f.write("====== MODEL B (WITH ELEMENT FEATURES) ======\n\n")
    f.write("VALIDATION:\n")
    f.write(format_metrics_txt(resB_val))
    f.write("\n\nTEST:\n")
    f.write(format_metrics_txt(resB_test))
print("Saved metrics_B.txt")


# ============================================================
# Prediction helper
# ============================================================
import torch
def predict_all_embedded(trainer, X_emb):
    xb = torch.tensor(X_emb, dtype=torch.float32, device=trainer.device)
    for net in trainer.base_nets.values():
        net.eval()
    trainer.meta.eval()
    with torch.no_grad():
        base = torch.cat([trainer.base_nets[p](xb) for p in trainer.present_targets], dim=1)
        pred_std = (base + trainer.meta(base)).cpu().numpy()
    return pred_std * trainer.σ + trainer.μ


# ============================================================
# Index helper
# ============================================================
def indices_with_element(trainer, element: str, split="test", min_frac=1e-12):
    smap = {"train": trainer.tr_idx, "val": trainer.va_idx, "test": trainer.te_idx}
    idxs = smap[split]
    frac = trainer.composition_df.loc[idxs, element].to_numpy(float)
    return idxs[frac > float(min_frac)]


# ============================================================
# PLOTTING (unchanged)
# ============================================================
def plot_coeffs_A_vs_B(trA, trB, idxs, title_prefix, outdir, fname_prefix):
    if len(idxs) == 0:
        print(f"[WARN] No rows for {title_prefix}")
        return

    mask = trA.mask_all[idxs].astype(bool)
    y_true = trA.y_raw[idxs]
    predA = predict_all_embedded(trA, trA.X_embedded[idxs])
    predB = predict_all_embedded(trB, trB.X_embedded[idxs])

    TGT = trA.present_targets
    N = len(TGT)

    for start in range(0, N, MAX_TARGETS_PER_FIG):
        chunk = TGT[start:start+MAX_TARGETS_PER_FIG]
        rows = len(chunk)

        fig, axes = plt.subplots(rows, 2, figsize=(13, 4.0*rows))
        if rows == 1:
            axes = np.array([axes])

        for r, t in enumerate(chunk):
            axA = axes[r, 0]
            axB = axes[r, 1]
            j = trA.idx_map[t]
            m = mask[:, j]
            if not np.any(m):
                axA.set_axis_off(); axB.set_axis_off()
                continue

            x = y_true[m, j]
            yA = predA[m, j]
            yB = predB[m, j]

            mn = min(x.min(), yA.min()); mx = max(x.max(), yA.max())
            axA.scatter(x, yA, alpha=0.65)
            axA.plot([mn, mx], [mn, mx], "k--")
            axA.set_title(f"{t} | A")
            axA.grid(True, alpha=0.25)

            mn = min(x.min(), yB.min()); mx = max(x.max(), yB.max())
            axB.scatter(x, yB, alpha=0.65)
            axB.plot([mn, mx], [mn, mx], "k--")
            axB.set_title(f"{t} | B")
            axB.grid(True, alpha=0.25)

        fig.suptitle(f"{title_prefix} — A vs B", fontsize=14)
        fig.tight_layout()
        fname = f"{fname_prefix}_A_vs_B_{start}_{start+rows-1}.png"
        fig.savefig(os.path.join(outdir, fname), dpi=160)
        plt.close(fig)
        print("Saved:", fname)


def plot_derived_A_vs_B(trA, trB, idxs, title_prefix, outdir, fname, T):
    props = ["rho", "muA", "muB", "k", "cp"]
    if len(idxs) == 0:
        print(f"[WARN] No rows for {title_prefix}")
        return

    predA = predict_all_embedded(trA, trA.X_embedded[idxs])
    predB = predict_all_embedded(trB, trB.X_embedded[idxs])

    true_vals = {p: [] for p in props}
    A_vals = {p: [] for p in props}
    B_vals = {p: [] for p in props}

    for pos, idx in enumerate(idxs):
        row = trA.df.iloc[idx]
        mask_row = trA.mask_all[idx]

        coeff_true = {}
        for j, col in enumerate(trA.present_targets):
            if mask_row[j]:
                coeff_true[col] = float(row[col])

        d_true = trA.derived(coeff_true, T)
        dA = trA.derived(dict(zip(trA.present_targets, predA[pos])), T)
        dB = trA.derived(dict(zip(trA.present_targets, predB[pos])), T)

        for p in props:
            a = d_true.get(p)
            va = dA.get(p)
            vb = dB.get(p)
            if a is None or not np.isfinite(a) or abs(a) <= 1e-12:
                continue
            if np.isfinite(va) and np.isfinite(vb):
                true_vals[p].append(a)
                A_vals[p].append(va)
                B_vals[p].append(vb)

    rows = len(props)
    fig, axes = plt.subplots(rows, 2, figsize=(13, 4.3*rows))
    for r, p in enumerate(props):
        axA, axB = axes[r]

        if len(true_vals[p]) == 0:
            axA.set_axis_off(); axB.set_axis_off(); continue

        x = np.array(true_vals[p])
        yA = np.array(A_vals[p])
        yB = np.array(B_vals[p])

        mn = min(x.min(), yA.min()); mx = max(x.max(), yA.max())
        axA.scatter(x, yA, alpha=0.65)
        axA.plot([mn, mx], [mn, mx], "k--")
        axA.set_title(f"{p} | A")
        axA.grid(True, alpha=0.25)

        mn = min(x.min(), yB.min()); mx = max(x.max(), yB.max())
        axB.scatter(x, yB, alpha=0.65)
        axB.plot([mn, mx], [mn, mx], "k--")
        axB.set_title(f"{p} | B")
        axB.grid(True, alpha=0.25)

    fig.suptitle(f"{title_prefix} — Derived, A vs B", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=160)
    plt.close(fig)
    print("Saved:", fname)


def plot_single_composition_A_vs_B(trA, trB, name, comp, outdir, T):
    coeffA = trA.predict(comp)
    coeffB = trB.predict(comp)

    targets = trA.present_targets
    N = len(targets)

    for start in range(0, N, MAX_TARGETS_PER_FIG):
        chunk = targets[start:start+MAX_TARGETS_PER_FIG]
        rows = len(chunk)

        fig, axes = plt.subplots(rows, 1, figsize=(10, 3.0*rows))
        if rows == 1:
            axes = np.array([axes])

        for r, t in enumerate(chunk):
            ax = axes[r]
            ax.bar(["A", "B"], [coeffA.get(t, np.nan), coeffB.get(t, np.nan)], alpha=0.85)
            ax.set_title(t)
            ax.grid(True, axis="y", alpha=0.25)

        fig.suptitle(f"Composition {name} — A vs B", fontsize=14)
        fig.tight_layout()
        fname = f"single_{name.replace(' ','_')}_A_vs_B_{start}_{start+rows-1}.png"
        fig.savefig(os.path.join(outdir, fname), dpi=160)
        plt.close(fig)
        print("Saved:", fname)


# ============================================================
# RUN PLOTS
# ============================================================
print(f"\nSaving comparison plots to: {PLOT_DIR}")

for el in ELEMENT_FILTERS:
    idxs = indices_with_element(trainerA, el, "test")
    title = f"Element {el} > 0 | test N={len(idxs)}"
    plot_coeffs_A_vs_B(trainerA, trainerB, idxs, title, PLOT_DIR, f"{el}_coeffs")
    plot_derived_A_vs_B(trainerA, trainerB, idxs, title, PLOT_DIR,
                        f"{el}_derived_T{TEMPERATURE}.png", TEMPERATURE)

for name, comp in SINGLE_COMPOSITIONS.items():
    plot_single_composition_A_vs_B(trainerA, trainerB, name, comp, PLOT_DIR, TEMPERATURE)

print("\nDONE.")
print("Plots →", PLOT_DIR)
print("Models A →", MODELS_A_DIR)
print("Models B →", MODELS_B_DIR)
print("Metric files →", os.path.join(BASE_OUTDIR, "metrics_A.txt"),
                     os.path.join(BASE_OUTDIR, "metrics_B.txt"))
