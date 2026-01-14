import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split

# Local import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processing_saltdblean.processor import SALTDBLEANProcessor
from processing_saltdblean.snn_trainer import (
    SNNMetaTrainer,
    TARGETS,
    DERIVED_PROPS,
    ELEMENT_FEATURE_COLS,
)

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz_with_ionic_polarizability.csv"

BASE_OUTDIR = os.path.join("data", "snn_compare")
PLOT_DIR = os.path.join(BASE_OUTDIR, "plots")
MODELS_A_DIR = os.path.join(BASE_OUTDIR, "models_A_without_elements")
MODELS_B_DIR = os.path.join(BASE_OUTDIR, "models_B_with_elements")

TEMPERATURE = 900
MAX_TARGETS_PER_FIG = 8
ELEMENT_FILTERS = ["Cl", "F"]

SINGLE_COMPOSITIONS = {
    "NaCl (50-50 atoms)": {"Na": 0.5, "Cl": 0.5},
}

# PCA settings (your choice)
EMBEDDING_METHOD = "pca"
N_COMPONENTS = 32

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODELS_A_DIR, exist_ok=True)
os.makedirs(MODELS_B_DIR, exist_ok=True)

# ============================================================
# Load and preprocess data
# ============================================================
processor = SALTDBLEANProcessor.from_csv(DATA_PATH)
processor.df.columns = processor.df.columns.str.strip()
df = processor.df

# ============================================================
# IMPORTANT: create ONE shared split for BOTH runs
# ============================================================
idx_all = np.arange(len(df))
tr_idx, te_idx = train_test_split(idx_all, test_size=0.20, random_state=SEED)
tr_idx, va_idx = train_test_split(tr_idx, test_size=0.20, random_state=SEED)
splits = (tr_idx, va_idx, te_idx)

# ============================================================
# Build TWO trainers: A (WITHOUT) and B (WITH)
# ============================================================
trainerA = SNNMetaTrainer(
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

trainerB = SNNMetaTrainer(
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

assert trainerA.present_targets == trainerB.present_targets, "Targets mismatch between A and B!"

# ============================================================
# Train A then B
# ============================================================
print("\n==============================")
print("TRAIN A (SNN): WITHOUT element features")
print("==============================")
trainerA.train_base()
trainerA.train_meta()
trainerA.evaluate(split="val")
trainerA.evaluate(split="test")

print("\n==============================")
print("TRAIN B (SNN): WITH element features")
print("==============================")
trainerB.train_base()
trainerB.train_meta()
trainerB.evaluate(split="val")
trainerB.evaluate(split="test")

# ============================================================
# Helper: predict for batches of embedded FEATURES (generic)
# ============================================================
def predict_all_embedded(trainer, X_embedded: np.ndarray) -> np.ndarray:
    for net in trainer.base_nets.values():
        net.eval()
    trainer.meta.eval()

    with torch.no_grad():
        xb = torch.tensor(X_embedded, dtype=torch.float32, device=trainer.device)
        base_out = torch.cat([trainer.base_nets[p](xb) for p in trainer.present_targets], dim=1)
        pred_std = (base_out + trainer.meta(base_out)).cpu().numpy()
        return pred_std * trainer.σ + trainer.μ

# ============================================================
# Index helper
# ============================================================
def indices_with_element(trainer, element: str, split: str = "test", min_frac: float = 1e-12) -> np.ndarray:
    split_map = {"train": trainer.tr_idx, "val": trainer.va_idx, "test": trainer.te_idx}
    idxs = split_map[split]
    frac = trainer.composition_df.loc[idxs, element].to_numpy(dtype=float)
    return idxs[frac > float(min_frac)]

# ============================================================
# Plotters: A vs B (2 columns)
# ============================================================
def plot_coeffs_A_vs_B(trA, trB, idxs, title_prefix, outdir, fname_prefix, max_targets_per_fig=8):
    if len(idxs) == 0:
        print(f"[WARN] No rows for: {title_prefix}. Skipping.")
        return

    y_true = trA.y_raw[idxs]
    mask = trA.mask_all[idxs].astype(bool)

    predA = predict_all_embedded(trA, trA.X_embedded[idxs])
    predB = predict_all_embedded(trB, trB.X_embedded[idxs])

    targets = trA.present_targets
    n = len(targets)

    for start in range(0, n, max_targets_per_fig):
        chunk = targets[start:start+max_targets_per_fig]
        rows = len(chunk)

        fig, axes = plt.subplots(rows, 2, figsize=(13, 4.2 * rows))
        if rows == 1:
            axes = np.array([axes])

        for r, t in enumerate(chunk):
            j = trA.idx_map[t]
            m = mask[:, j]

            axA = axes[r, 0]
            axB = axes[r, 1]

            if not np.any(m):
                axA.set_axis_off(); axB.set_axis_off()
                continue

            x = y_true[m, j]
            yA = predA[m, j]
            yB = predB[m, j]

            axA.scatter(x, yA, alpha=0.65)
            mn = min(x.min(), yA.min()); mx = max(x.max(), yA.max())
            axA.plot([mn, mx], [mn, mx], "k--", linewidth=1)
            axA.set_title(f"{t} | WITHOUT elem-feats")
            axA.set_xlabel("Actual"); axA.set_ylabel("Predicted")
            axA.grid(True, alpha=0.25)

            axB.scatter(x, yB, alpha=0.65)
            mn = min(x.min(), yB.min()); mx = max(x.max(), yB.max())
            axB.plot([mn, mx], [mn, mx], "k--", linewidth=1)
            axB.set_title(f"{t} | WITH elem-feats")
            axB.set_xlabel("Actual"); axB.set_ylabel("Predicted")
            axB.grid(True, alpha=0.25)

        fig.suptitle(f"{title_prefix}\nCoefficients: Actual vs Predicted — A vs B", y=1.01, fontsize=14)
        fig.tight_layout()

        fname = f"{fname_prefix}_coeff_A_vs_B_{start}_{start+len(chunk)-1}.png"
        fig.savefig(os.path.join(outdir, fname), dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}")

def plot_derived_A_vs_B(trA, trB, idxs, title_prefix, outdir, fname, T, props=("rho","muA","muB","k","cp")):
    if len(idxs) == 0:
        print(f"[WARN] No rows for: {title_prefix}. Skipping.")
        return

    predA = predict_all_embedded(trA, trA.X_embedded[idxs])
    predB = predict_all_embedded(trB, trB.X_embedded[idxs])

    actual_vals = {p: [] for p in props}
    predA_vals = {p: [] for p in props}
    predB_vals = {p: [] for p in props}

    for kpos, idx in enumerate(idxs):
        row = trA.df.iloc[idx]
        mask_row = trA.mask_all[idx]

        actual_coeffs = {}
        for j, col in enumerate(trA.present_targets):
            if mask_row[j]:
                actual_coeffs[col] = float(row[col])

        aprops = trA.derived(actual_coeffs, T)

        coeffA = dict(zip(trA.present_targets, predA[kpos]))
        coeffB = dict(zip(trB.present_targets, predB[kpos]))
        pA = trA.derived(coeffA, T)
        pB = trB.derived(coeffB, T)

        for p in props:
            a = aprops.get(p)
            va = pA.get(p)
            vb = pB.get(p)
            if a is None or va is None or vb is None:
                continue
            if not (np.isfinite(a) and np.isfinite(va) and np.isfinite(vb)):
                continue
            if abs(a) <= 1e-12:
                continue
            actual_vals[p].append(a)
            predA_vals[p].append(va)
            predB_vals[p].append(vb)

    rows = len(props)
    fig, axes = plt.subplots(rows, 2, figsize=(13, 4.0 * rows))
    if rows == 1:
        axes = np.array([axes])

    for r, p in enumerate(props):
        axA = axes[r, 0]
        axB = axes[r, 1]

        if len(actual_vals[p]) == 0:
            axA.set_axis_off(); axB.set_axis_off()
            continue

        x = np.array(actual_vals[p], float)
        yA = np.array(predA_vals[p], float)
        yB = np.array(predB_vals[p], float)

        axA.scatter(x, yA, alpha=0.65)
        mn = min(x.min(), yA.min()); mx = max(x.max(), yA.max())
        axA.plot([mn, mx], [mn, mx], "k--", linewidth=1)
        axA.set_title(f"{p} @ {T}K | WITHOUT elem-feats")
        axA.set_xlabel("Actual"); axA.set_ylabel("Predicted")
        axA.grid(True, alpha=0.25)

        axB.scatter(x, yB, alpha=0.65)
        mn = min(x.min(), yB.min()); mx = max(x.max(), yB.max())
        axB.plot([mn, mx], [mn, mx], "k--", linewidth=1)
        axB.set_title(f"{p} @ {T}K | WITH elem-feats")
        axB.set_xlabel("Actual"); axB.set_ylabel("Predicted")
        axB.grid(True, alpha=0.25)

    fig.suptitle(f"{title_prefix}\nDerived properties: Actual vs Predicted — A vs B", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")

def plot_single_composition_A_vs_B(trA, trB, name, comp, outdir, T):
    coeffA = trA.predict(comp)
    coeffB = trB.predict(comp)

    targets = trA.present_targets
    n = len(targets)

    for start in range(0, n, MAX_TARGETS_PER_FIG):
        chunk = targets[start:start+MAX_TARGETS_PER_FIG]
        rows = len(chunk)

        fig, axes = plt.subplots(rows, 1, figsize=(11, 3.2 * rows))
        if rows == 1:
            axes = np.array([axes])

        for r, t in enumerate(chunk):
            ax = axes[r]
            a = coeffA.get(t, np.nan)
            b = coeffB.get(t, np.nan)
            ax.bar(["WITHOUT elem-feats", "WITH elem-feats"], [a, b], alpha=0.85)
            ax.set_title(t)
            ax.grid(True, axis="y", alpha=0.25)

        fig.suptitle(f"Single composition: {name}\nCoefficients — A vs B", y=1.01, fontsize=14)
        fig.tight_layout()

        fname = f"single_{name.replace(' ', '_')}_coeffs_A_vs_B_{start}_{start+len(chunk)-1}.png"
        fig.savefig(os.path.join(outdir, fname), dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}")

# ============================================================
# RUN PLOTS
# ============================================================
print(f"\nSaving comparison plots under: {PLOT_DIR}")

for el in ELEMENT_FILTERS:
    idxs = indices_with_element(trainerA, element=el, split="test")
    title_prefix = f"Element filter: {el} > 0 | split=test | N={len(idxs)}"
    fname_prefix = f"{el}_test"
    plot_coeffs_A_vs_B(trainerA, trainerB, idxs, title_prefix, PLOT_DIR, fname_prefix, MAX_TARGETS_PER_FIG)
    plot_derived_A_vs_B(trainerA, trainerB, idxs, title_prefix, PLOT_DIR, f"{el}_derived_A_vs_B_T{int(TEMPERATURE)}K.png", TEMPERATURE)

for name, comp in SINGLE_COMPOSITIONS.items():
    plot_single_composition_A_vs_B(trainerA, trainerB, name, comp, PLOT_DIR, TEMPERATURE)

print("\nDone.")
print("Plots:", PLOT_DIR)
print("Models A:", MODELS_A_DIR)
print("Models B:", MODELS_B_DIR)
