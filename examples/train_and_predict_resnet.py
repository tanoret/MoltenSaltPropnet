import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Local import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processing_saltdblean.processor import SALTDBLEANProcessor
from processing_saltdblean.resnet_trainer import (
    ResNetMetaTrainer,
    TARGETS,
    DERIVED_PROPS,
    ELEMENT_FEATURE_COLS,
)

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz_with_ionic_polarizability.csv"

BASE_OUTDIR = os.path.join("data", "resnet")
PLOT_DIR = os.path.join(BASE_OUTDIR, "plots")

TEMPERATURE = 900
MAX_TARGETS_PER_FIG = 8

# Filters to generate "element-containing" plots (any salt containing the element)
ELEMENT_FILTERS = ["Cl", "F"]

# Single-composition prediction plots
SINGLE_COMPOSITIONS = {
    "NaCl (50-50 atoms)": {"Na": 0.5, "Cl": 0.5},
}

os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# Load and preprocess data
# ============================================================
processor = SALTDBLEANProcessor.from_csv(DATA_PATH)
processor.df.columns = processor.df.columns.str.strip()

# Instantiate trainer (element features are included by default via ELEMENT_FEATURE_COLS)
trainer = ResNetMetaTrainer(processor.df, TARGETS, DERIVED_PROPS, ELEMENT_FEATURE_COLS)

# ============================================================
# Train
# ============================================================
trainer.train_base()
trainer.train_meta()

# ============================================================
# Helper: predict for batches of embedded FEATURES
# ============================================================
def predict_all_embedded(X_embedded: np.ndarray) -> np.ndarray:
    """
    X_embedded must be in the same space the nets were trained on (trainer.X_embedded).
    Returns raw-scale predictions: shape (N, n_targets).
    """
    for net in trainer.base_nets.values():
        net.eval()
    trainer.meta.eval()

    with torch.no_grad():
        xb = torch.tensor(X_embedded, dtype=torch.float32, device=trainer.device)
        base_out = torch.stack([trainer.base_nets[p](xb) for p in trainer.present_targets], dim=1)
        pred_std = (base_out + trainer.meta(base_out)).cpu().numpy()
        return pred_std * trainer.σ + trainer.μ


# ============================================================
# Index helper: element-containing filter (train/val/test)
# ============================================================
def indices_with_element(trainer, element: str, split: str = "test", min_frac: float = 1e-12) -> np.ndarray:
    split_map = {"train": trainer.tr_idx, "val": trainer.va_idx, "test": trainer.te_idx}
    if split not in split_map:
        raise ValueError(f"split must be one of {list(split_map.keys())}")

    if not hasattr(trainer, "composition_df"):
        raise AttributeError("trainer must have composition_df")
    if element not in trainer.composition_df.columns:
        raise ValueError(
            f"Element '{element}' not present in composition_df. "
            f"Sample cols: {list(trainer.composition_df.columns)[:15]} ..."
        )

    idxs = split_map[split]
    frac = trainer.composition_df.loc[idxs, element].to_numpy(dtype=float)
    return idxs[frac > float(min_frac)]


# ============================================================
# Plotters: coefficient actual vs pred (masked) for a set of indices
# ============================================================
def plot_coeffs_actual_vs_pred_subplots_for_indices(
    trainer,
    idxs: np.ndarray,
    title_prefix: str,
    outdir: str,
    fname_prefix: str,
    max_targets_per_fig: int = 8,
):
    if len(idxs) == 0:
        print(f"[WARN] No rows for: {title_prefix}. Skipping coefficient plots.")
        return

    y_true = trainer.y_raw[idxs]                       # (N, n_targets), raw
    mask = trainer.mask_all[idxs].astype(bool)         # (N, n_targets), True where real
    y_pred = predict_all_embedded(trainer.X_embedded[idxs])

    targets = trainer.present_targets
    n = len(targets)

    for start in range(0, n, max_targets_per_fig):
        chunk = targets[start : start + max_targets_per_fig]
        rows = len(chunk)

        fig, axes = plt.subplots(rows, 1, figsize=(8.5, 4.2 * rows))
        if rows == 1:
            axes = np.array([axes])

        for r, t in enumerate(chunk):
            j = trainer.idx_map[t]
            m = mask[:, j]
            ax = axes[r]

            if not np.any(m):
                ax.set_axis_off()
                continue

            x = y_true[m, j]
            y = y_pred[m, j]

            ax.scatter(x, y, alpha=0.65)
            mn = min(x.min(), y.min())
            mx = max(x.max(), y.max())
            ax.plot([mn, mx], [mn, mx], "k--", linewidth=1, label="y=x")

            ax.set_title(f"{t}")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")

        fig.suptitle(
            f"{title_prefix}\nActual vs Predicted coefficients",
            y=1.01,
            fontsize=14,
        )
        fig.tight_layout()

        fname = f"{fname_prefix}_coeff_actual_vs_pred_targets_{start}_{start+len(chunk)-1}.png"
        fig.savefig(os.path.join(outdir, fname), dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}")


# ============================================================
# Plotters: derived actual vs pred (masked) for a set of indices
# ============================================================
def plot_derived_actual_vs_pred_for_indices(
    trainer,
    idxs: np.ndarray,
    title_prefix: str,
    outdir: str,
    fname: str,
    T: float,
    props=("rho", "muA", "muB", "k", "cp"),
):
    if len(idxs) == 0:
        print(f"[WARN] No rows for: {title_prefix}. Skipping derived plots.")
        return

    pred = predict_all_embedded(trainer.X_embedded[idxs])

    actual_vals = {p: [] for p in props}
    pred_vals = {p: [] for p in props}

    for kpos, idx in enumerate(idxs):
        row = trainer.df.iloc[idx]
        mask_row = trainer.mask_all[idx]  # bool for present_targets

        # Actual coeffs only where present for that row
        actual_coeffs = {}
        for j, col in enumerate(trainer.present_targets):
            if mask_row[j]:
                actual_coeffs[col] = float(row[col])

        aprops = trainer.derived(actual_coeffs, T)

        pred_coeffs = dict(zip(trainer.present_targets, pred[kpos]))
        pprops = trainer.derived(pred_coeffs, T)

        for p in props:
            a = aprops.get(p)
            phat = pprops.get(p)
            if a is None or phat is None:
                continue
            if not (np.isfinite(a) and np.isfinite(phat)):
                continue
            if abs(a) <= 1e-12:
                continue
            actual_vals[p].append(a)
            pred_vals[p].append(phat)

    rows = len(props)
    fig, axes = plt.subplots(rows, 1, figsize=(8.5, 4.0 * rows))
    if rows == 1:
        axes = np.array([axes])

    for r, p in enumerate(props):
        ax = axes[r]
        if len(actual_vals[p]) == 0:
            ax.set_axis_off()
            continue

        x = np.array(actual_vals[p], dtype=float)
        y = np.array(pred_vals[p], dtype=float)

        ax.scatter(x, y, alpha=0.65)
        mn = min(x.min(), y.min())
        mx = max(x.max(), y.max())
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1, label="y=x")
        ax.set_title(f"{p} @ {T}K")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    fig.suptitle(
        f"{title_prefix}\nDerived properties Actual vs Predicted @ {T}K",
        y=1.01,
        fontsize=14,
    )
    fig.tight_layout()

    fig.savefig(os.path.join(outdir, fname), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# ============================================================
# Plotters: single-composition prediction (coeffs + derived)
# ============================================================
def plot_single_composition_coeffs_and_derived(trainer, name: str, composition: dict, outdir: str, T: float):
    pred_coeffs = trainer.predict(composition)

    # ---- Coefficients figure: chunked into multiple pages if many targets
    targets = trainer.present_targets
    n = len(targets)

    for start in range(0, n, MAX_TARGETS_PER_FIG):
        chunk = targets[start : start + MAX_TARGETS_PER_FIG]
        rows = len(chunk)

        fig, axes = plt.subplots(rows, 1, figsize=(10, 3.0 * rows))
        if rows == 1:
            axes = np.array([axes])

        for r, t in enumerate(chunk):
            ax = axes[r]
            v = pred_coeffs.get(t, np.nan)

            ax.bar([t], [v], alpha=0.85)
            ax.set_ylabel("Predicted")
            ax.grid(True, axis="y", alpha=0.25)

        fig.suptitle(
            f"Single composition prediction: {name}\nCoefficients",
            y=1.01,
            fontsize=14,
        )
        fig.tight_layout()

        fname = f"single_{name.replace(' ', '_')}_coeffs_{start}_{start+len(chunk)-1}.png"
        fig.savefig(os.path.join(outdir, fname), dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}")

    # ---- Derived properties figure (one page)
    props = ["rho", "muA", "muB", "k", "cp"]
    derived_vals = trainer.derived(pred_coeffs, T)

    fig, axes = plt.subplots(len(props), 1, figsize=(10, 2.8 * len(props)))
    if len(props) == 1:
        axes = np.array([axes])

    for r, p in enumerate(props):
        ax = axes[r]
        v = derived_vals.get(p, np.nan)

        ax.bar([p], [v], alpha=0.85)
        ax.set_title(f"{p} @ {T}K")
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"Single composition prediction: {name}\nDerived properties @ {T}K",
        y=1.01,
        fontsize=14,
    )
    fig.tight_layout()

    fname = f"single_{name.replace(' ', '_')}_derived_T{int(T)}K.png"
    fig.savefig(os.path.join(outdir, fname), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# ============================================================
# Run
# ============================================================
print(f"\nSaving plots under: {PLOT_DIR}")

# Element-filtered plots (test split)
for el in ELEMENT_FILTERS:
    idxs = indices_with_element(trainer, element=el, split="test")
    title_prefix = f"Element filter: {el} > 0 (any {el}-containing salt) | split=test | N={len(idxs)}"
    fname_prefix = f"{el}_contains_test"

    plot_coeffs_actual_vs_pred_subplots_for_indices(
        trainer,
        idxs=idxs,
        title_prefix=title_prefix,
        outdir=PLOT_DIR,
        fname_prefix=fname_prefix,
        max_targets_per_fig=MAX_TARGETS_PER_FIG,
    )

    plot_derived_actual_vs_pred_for_indices(
        trainer,
        idxs=idxs,
        title_prefix=title_prefix,
        outdir=PLOT_DIR,
        fname=f"{el}_contains_test_derived_actual_vs_pred_T{int(TEMPERATURE)}K.png",
        T=TEMPERATURE,
    )

# Single-composition plots
for name, comp in SINGLE_COMPOSITIONS.items():
    plot_single_composition_coeffs_and_derived(
        trainer,
        name=name,
        composition=comp,
        outdir=PLOT_DIR,
        T=TEMPERATURE,
    )

print("\nDone. Plots written to:", PLOT_DIR)
