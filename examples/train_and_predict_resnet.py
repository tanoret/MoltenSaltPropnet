import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
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

# Everything will be stored under: data/resnet/...
BASE_OUTDIR = os.path.join("data", "resnet")
PLOT_DIR = os.path.join(BASE_OUTDIR, "plots")

TEMPERATURE = 900
MAX_TARGETS_PER_FIG = 8

# Filters to generate "element-only" subplot figures (any salt containing the element)
ELEMENT_FILTERS = ["Cl", "F"]

# Single-composition prediction plots (not dataset-filtered)
SINGLE_COMPOSITIONS = {
    "NaCl (50-50 atoms)": {"Na": 0.5, "Cl": 0.5},
}

os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# Load and preprocess data
# ============================================================
processor = SALTDBLEANProcessor.from_csv(DATA_PATH)
processor.df.columns = processor.df.columns.str.strip()

trainer = ResNetMetaTrainer(processor.df, TARGETS, DERIVED_PROPS, ELEMENT_FEATURE_COLS)

# ============================================================
# Train
# ============================================================
trainer.train_base()
trainer.train_meta()

# ============================================================
# Helper: predict for batches of embedded FEATURES
# ============================================================
def predict_all(X_input: np.ndarray) -> np.ndarray:
    for net in trainer.base_nets.values():
        net.eval()
    trainer.meta.eval()

    with torch.no_grad():
        xb = torch.tensor(X_input, dtype=torch.float32, device=trainer.device)
        base_out = torch.stack([trainer.base_nets[p](xb) for p in trainer.present_targets], 1)
        pred_std = (base_out + trainer.meta(base_out)).cpu().numpy()
        return pred_std * trainer.σ + trainer.μ


# ============================================================
# Script-only helpers (NO changes inside ResNetMetaTrainer)
# ============================================================
def embedded_features_without_element_features(trainer, idxs: np.ndarray) -> np.ndarray:
    """
    Build embedded feature matrix for rows idxs, but zero-out last k element-feature columns
    BEFORE embedding. If embedding_method == 'none', returns raw features.
    """
    X_raw = trainer.X[idxs].copy()
    k = len(ELEMENT_FEATURE_COLS)
    X_raw[:, -k:] = 0.0

    if getattr(trainer, "embedding_method", "none") != "none":
        return trainer.embedder.transform(X_raw)
    else:
        return X_raw


def indices_with_element(trainer, element: str, split: str = "test", min_frac: float = 1e-12) -> np.ndarray:
    """
    'Any X-containing salt' filter: returns indices in split where element fraction > min_frac.
    """
    split_map = {"train": trainer.tr_idx, "val": trainer.va_idx, "test": trainer.te_idx}
    if split not in split_map:
        raise ValueError(f"split must be one of {list(split_map.keys())}")

    if not hasattr(trainer, "composition_df"):
        raise AttributeError("trainer must have composition_df")

    if element not in trainer.composition_df.columns:
        raise ValueError(
            f"Element '{element}' not present in composition_df. Sample cols: {list(trainer.composition_df.columns)[:15]} ..."
        )

    idxs = split_map[split]
    frac = trainer.composition_df.loc[idxs, element].to_numpy(dtype=float)
    return idxs[frac > float(min_frac)]


# ============================================================
# Plotters: element-filtered (Cl-only, F-only) as subplots WITH vs WITHOUT
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

    y_true = trainer.y_raw[idxs]
    mask = trainer.mask_all[idxs].astype(bool)

    y_pred_with = predict_all(trainer.X_embedded[idxs])
    y_pred_wo = predict_all(embedded_features_without_element_features(trainer, idxs))

    targets = trainer.present_targets
    n = len(targets)

    for start in range(0, n, max_targets_per_fig):
        chunk = targets[start : start + max_targets_per_fig]
        rows = len(chunk)
        cols = 2  # WITH | WITHOUT

        fig, axes = plt.subplots(rows, cols, figsize=(13, 4.2 * rows))
        if rows == 1:
            axes = np.array([axes])

        for r, t in enumerate(chunk):
            j = trainer.idx_map[t]
            m = mask[:, j]

            ax_with = axes[r, 0]
            ax_wo = axes[r, 1]

            if not np.any(m):
                ax_with.set_axis_off()
                ax_wo.set_axis_off()
                continue

            x = y_true[m, j]
            yW = y_pred_with[m, j]
            yN = y_pred_wo[m, j]

            # WITH (orange)
            ax_with.scatter(x, yW, alpha=0.65, label="WITH element features", color="orange")
            mn = min(x.min(), yW.min())
            mx = max(x.max(), yW.max())
            ax_with.plot([mn, mx], [mn, mx], "k--", linewidth=1, label="y=x")
            ax_with.set_title(f"{t} | WITH")
            ax_with.set_xlabel("Actual")
            ax_with.set_ylabel("Predicted")
            ax_with.grid(True, alpha=0.25)
            ax_with.legend(loc="best")

            # WITHOUT (blue)
            ax_wo.scatter(x, yN, alpha=0.65, label="WITHOUT element features", color="tab:blue")
            mn2 = min(x.min(), yN.min())
            mx2 = max(x.max(), yN.max())
            ax_wo.plot([mn2, mx2], [mn2, mx2], "k--", linewidth=1, label="y=x")
            ax_wo.set_title(f"{t} | WITHOUT")
            ax_wo.set_xlabel("Actual")
            ax_wo.set_ylabel("Predicted")
            ax_wo.grid(True, alpha=0.25)
            ax_wo.legend(loc="best")

        fig.suptitle(
            f"{title_prefix}\nActual vs Predicted coefficients (subplots) — WITH vs WITHOUT element features",
            y=1.01,
            fontsize=14,
        )
        fig.tight_layout()

        fname = f"{fname_prefix}_coeff_actual_vs_pred_targets_{start}_{start+len(chunk)-1}.png"
        fig.savefig(os.path.join(outdir, fname), dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}")


def plot_derived_actual_vs_pred_subplots_for_indices(
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

    pred_with = predict_all(trainer.X_embedded[idxs])
    pred_wo = predict_all(embedded_features_without_element_features(trainer, idxs))

    actual_vals = {p: [] for p in props}
    pred_vals_with = {p: [] for p in props}
    pred_vals_wo = {p: [] for p in props}

    for kpos, idx in enumerate(idxs):
        row = trainer.df.iloc[idx]
        mask_row = trainer.mask_all[idx]

        # Actual coeffs only where present for that row
        actual_coeffs = {}
        for j, col in enumerate(trainer.present_targets):
            if mask_row[j]:
                actual_coeffs[col] = float(row[col])

        aprops = trainer.derived(actual_coeffs, T)

        pred_coeffs_with = dict(zip(trainer.present_targets, pred_with[kpos]))
        pred_coeffs_wo = dict(zip(trainer.present_targets, pred_wo[kpos]))
        pprops_with = trainer.derived(pred_coeffs_with, T)
        pprops_wo = trainer.derived(pred_coeffs_wo, T)

        for p in props:
            a = aprops.get(p)
            pw = pprops_with.get(p)
            pn = pprops_wo.get(p)
            if a is None or pw is None or pn is None:
                continue
            if not (np.isfinite(a) and np.isfinite(pw) and np.isfinite(pn)):
                continue
            if abs(a) <= 1e-12:
                continue

            actual_vals[p].append(a)
            pred_vals_with[p].append(pw)
            pred_vals_wo[p].append(pn)

    rows = len(props)
    fig, axes = plt.subplots(rows, 2, figsize=(13, 4.0 * rows))
    if rows == 1:
        axes = np.array([axes])

    for r, p in enumerate(props):
        ax_with = axes[r, 0]
        ax_wo = axes[r, 1]

        if len(actual_vals[p]) == 0:
            ax_with.set_axis_off()
            ax_wo.set_axis_off()
            continue

        x = np.array(actual_vals[p], dtype=float)
        yW = np.array(pred_vals_with[p], dtype=float)
        yN = np.array(pred_vals_wo[p], dtype=float)

        # WITH (orange)
        ax_with.scatter(x, yW, alpha=0.65, label="WITH element features", color="orange")
        mn = min(x.min(), yW.min())
        mx = max(x.max(), yW.max())
        ax_with.plot([mn, mx], [mn, mx], "k--", linewidth=1, label="y=x")
        ax_with.set_title(f"{p} @ {T}K | WITH")
        ax_with.set_xlabel("Actual")
        ax_with.set_ylabel("Predicted")
        ax_with.grid(True, alpha=0.25)
        ax_with.legend(loc="best")

        # WITHOUT (blue)
        ax_wo.scatter(x, yN, alpha=0.65, label="WITHOUT element features", color="tab:blue")
        mn2 = min(x.min(), yN.min())
        mx2 = max(x.max(), yN.max())
        ax_wo.plot([mn2, mx2], [mn2, mx2], "k--", linewidth=1, label="y=x")
        ax_wo.set_title(f"{p} @ {T}K | WITHOUT")
        ax_wo.set_xlabel("Actual")
        ax_wo.set_ylabel("Predicted")
        ax_wo.grid(True, alpha=0.25)
        ax_wo.legend(loc="best")

    fig.suptitle(
        f"{title_prefix}\nDerived properties Actual vs Predicted @ {T}K — WITH vs WITHOUT element features",
        y=1.01,
        fontsize=14,
    )
    fig.tight_layout()

    fig.savefig(os.path.join(outdir, fname), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# ============================================================
# Plotters: single-composition prediction (NaCl) as grouped subplots
# ============================================================
def predict_single_coeffs_with_without(trainer, composition: dict) -> tuple[dict, dict]:
    """
    Returns:
      (pred_with_dict, pred_without_dict) for the same composition.
    'WITHOUT' means element-features portion is zeroed before inference.
    """
    # Build "WITH" via trainer.predict()
    pred_with = trainer.predict(composition)

    # Build the feature vector exactly like trainer.predict(), then zero out elem-feats.
    # We do this in a script-only way, without editing trainer.
    # ----------------------------------------------------------------
    # Convert input composition to element-only composition (same logic as trainer.predict)
    elements = {}
    for key, value in composition.items():
        parsed = trainer.parse_compound(key)
        for el, count in parsed.items():
            elements[el] = elements.get(el, 0.0) + float(value) * float(count)

    total = sum(elements.values())
    if total <= 0:
        raise ValueError("Composition must have positive total")
    normalized = {k: v / total for k, v in elements.items()}  # element-only

    # Fraction vector in training column order
    frac = np.zeros(len(trainer.X_comp.columns), dtype=np.float32)
    for i, col in enumerate(trainer.X_comp.columns):
        frac[i] = normalized.get(col, 0.0)

    # Poly features
    raw_df = pd.DataFrame([frac], columns=trainer.X_comp.columns).fillna(0.0)
    raw_poly = trainer.poly.transform(raw_df).astype(np.float32)
    raw_poly = trainer.poly_scaler.transform(raw_poly).astype(np.float32)

    # Element aggregated features (weighted mean) using lookup
    elem_vec = []
    for col in trainer.ELEMENT_FEATURE_COLS:
        prop_map = trainer.elem_lookup.get(col, {})
        s = 0.0
        for el, f in normalized.items():
            s += float(f) * float(prop_map.get(el, 0.0))
        elem_vec.append(s)
    elem_vec = np.array(elem_vec, dtype=np.float32)[None, :]
    elem_vec = trainer.elem_scaler.transform(elem_vec).astype(np.float32)

    # Assemble full feature vector and then zero-out elem-features for WITHOUT
    feats_with = np.hstack([raw_poly, frac[None, :], elem_vec]).astype(np.float32)
    feats_wo = feats_with.copy()
    k = len(trainer.ELEMENT_FEATURE_COLS)
    feats_wo[:, -k:] = 0.0

    # Embedding (if any)
    if getattr(trainer, "embedding_method", "none") != "none":
        feats_with = trainer.embedder.transform(feats_with)
        feats_wo = trainer.embedder.transform(feats_wo)

    # Forward pass
    for net in trainer.base_nets.values():
        net.eval()
    trainer.meta.eval()

    with torch.no_grad():
        xb_wo = torch.tensor(feats_wo, dtype=torch.float32, device=trainer.device)
        base_out_wo = torch.stack([trainer.base_nets[p](xb_wo) for p in trainer.present_targets], 1)
        pred_std_wo = (base_out_wo + trainer.meta(base_out_wo)).cpu().numpy()[0]
        pred_raw_wo = pred_std_wo * trainer.σ + trainer.μ

    pred_without = {prop: float(pred_raw_wo[i]) for i, prop in enumerate(trainer.present_targets)}
    return pred_with, pred_without


def plot_single_composition_coeffs_and_derived(trainer, name: str, composition: dict, outdir: str, T: float):
    pred_with, pred_wo = predict_single_coeffs_with_without(trainer, composition)

    # ---- Coefficients figure: chunked into multiple pages if many targets
    targets = trainer.present_targets
    n = len(targets)

    for start in range(0, n, MAX_TARGETS_PER_FIG):
        chunk = targets[start : start + MAX_TARGETS_PER_FIG]
        rows = len(chunk)

        fig, axes = plt.subplots(rows, 1, figsize=(12, 3.2 * rows))
        if rows == 1:
            axes = np.array([axes])

        for r, t in enumerate(chunk):
            ax = axes[r]
            w = pred_with.get(t, np.nan)
            nval = pred_wo.get(t, np.nan)

            # two bars
            ax.bar(["WITH elem-feats", "WITHOUT elem-feats"], [w, nval], color=["orange", "tab:blue"], alpha=0.85)
            ax.set_title(t)
            ax.grid(True, axis="y", alpha=0.25)

        fig.suptitle(
            f"Single composition prediction: {name}\nCoefficients — WITH vs WITHOUT element features",
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
    derived_with = trainer.derived(pred_with, T)
    derived_wo = trainer.derived(pred_wo, T)

    fig, axes = plt.subplots(len(props), 1, figsize=(12, 3.0 * len(props)))
    if len(props) == 1:
        axes = np.array([axes])

    for r, p in enumerate(props):
        ax = axes[r]
        w = derived_with.get(p, np.nan)
        nval = derived_wo.get(p, np.nan)
        ax.bar(["WITH elem-feats", "WITHOUT elem-feats"], [w, nval], color=["orange", "tab:blue"], alpha=0.85)
        ax.set_title(f"{p} @ {T}K")
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"Single composition prediction: {name}\nDerived properties @ {T}K — WITH vs WITHOUT element features",
        y=1.01,
        fontsize=14,
    )
    fig.tight_layout()

    fname = f"single_{name.replace(' ', '_')}_derived_T{int(T)}K.png"
    fig.savefig(os.path.join(outdir, fname), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# ============================================================
# Run: element-only chloride + fluorine plots (test split)
# ============================================================
print(f"\nSaving plots under: {PLOT_DIR}")

for el in ELEMENT_FILTERS:
    idxs = indices_with_element(trainer, element=el, split="test")
    title_prefix = f"Element filter: {el} > 0 (any {el}-containing salt) | split=test | N={len(idxs)}"
    fname_prefix = f"{el}_only_test"

    plot_coeffs_actual_vs_pred_subplots_for_indices(
        trainer,
        idxs=idxs,
        title_prefix=title_prefix,
        outdir=PLOT_DIR,
        fname_prefix=fname_prefix,
        max_targets_per_fig=MAX_TARGETS_PER_FIG,
    )

    plot_derived_actual_vs_pred_subplots_for_indices(
        trainer,
        idxs=idxs,
        title_prefix=title_prefix,
        outdir=PLOT_DIR,
        fname=f"{el}_only_test_derived_actual_vs_pred_T{int(TEMPERATURE)}K.png",
        T=TEMPERATURE,
    )

# ============================================================
# Run: single-composition NaCl prediction plots (WITH vs WITHOUT)
# ============================================================
for name, comp in SINGLE_COMPOSITIONS.items():
    plot_single_composition_coeffs_and_derived(
        trainer,
        name=name,
        composition=comp,
        outdir=PLOT_DIR,
        T=TEMPERATURE,
    )

print("\nDone. Plots written to:", PLOT_DIR)


"""
print("\nFeature Index to Name Mapping:")
for i, col in enumerate(processor.df.columns):
    print(f"{i}: {col}")


print("X shape:", trainer.X.shape)
print("Embedded:", trainer.X_embedded.shape)
print("Feature names:", len(trainer.feature_names))


plot_dir = "resnet_prediction_plots"
os.makedirs(plot_dir, exist_ok=True)

#print("feat_dim =", trainer.feat_dim)
#print("X shape  =", trainer.X.shape)


n_samples = min(50, len(trainer.tr_idx))
X_shap = trainer.X_embedded[trainer.tr_idx[:n_samples]]

print("Using", X_shap.shape[0], "samples for SHAP")

# ----------------------------------------
# 2. Define a SHAP-compatible predict()
# ----------------------------------------

def model_predict(X_numpy):
    X_tensor = torch.tensor(X_numpy, dtype=torch.float32, device=trainer.device)
    with torch.no_grad():
        base = torch.stack([
            trainer.base_nets[p](X_tensor) for p in trainer.present_targets
        ], dim=1)
        out = base + trainer.meta(base)
        preds = (out * trainer.σ + trainer.μ).cpu().numpy()
    return preds


# ----------------------------------------
# 3. Background (Shap uses reference samples)
# ----------------------------------------

background_size = min(20, len(X_shap))
background = X_shap[:background_size]

print("Background size:", background_size)


# ----------------------------------------
# 4. Kernel SHAP — works for ANY model
# ----------------------------------------

print("\nInitializing KernelExplainer...")
explainer = shap.KernelExplainer(model_predict, background)

print("Computing SHAP values (this may take a few minutes)...")
shap_vals = explainer.shap_values(X_shap)

# shap_vals is a list: one array per target
print("\nNumber of output heads:", len(shap_vals))
print("Shape of SHAP for first target:", shap_vals[0].shape)


# ----------------------------------------
# 5. Interpret SHAP values using REAL FEATURE NAMES
# ----------------------------------------

feature_names = trainer.feature_names      # Human-readable names
num_outputs = len(shap_vals)

top_k = 15

for target_idx in range(num_outputs):
    target_name = trainer.present_targets[target_idx]
    vals = shap_vals[target_idx]                      # shape: (n_samples, feat_dim)
    mean_abs = np.mean(np.abs(vals), axis=0)          # (feat_dim,)

    # Select top-k
    top_idx = np.argsort(mean_abs)[-top_k:][::-1]

    print(f"\nTop {top_k} features for: {target_name}")
    print("----------------------------------------------")
    for rank, fidx in enumerate(top_idx, start=1):
        fname = feature_names[fidx]
        print(f"{rank:2d}. {fname:50s}  SHAP = {mean_abs[fidx]:.6f}")

"""

"""

# SHAP GradientExplainer

print("\n SHAP values using GradientExplainer...")

n_samples = min(50, len(trainer.tr_idx))
#X_shap = trainer.X[trainer.tr_tr_idx[:50]]
X_shap = trainer.X_embedded[trainer.tr_idx[:n_samples]]
X_tensor = torch.tensor(X_shap, dtype=torch.float32, device=trainer.device)

# Wrap base + meta network for SHAP
class CombinedResNetModel(nn.Module):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def forward(self, x):
        base = torch.stack([self.trainer.base_nets[p](x) for p in self.trainer.present_targets], dim=1)
        return base + self.trainer.meta(base)

model = CombinedResNetModel(trainer).to(trainer.device)
model.eval()

# background sample (mean of features)
background = torch.tensor(
    X_shap.mean(axis=0, keepdims=True),
    dtype=torch.float32,
    device=trainer.device
)

#explainer = shap.KernelExplainer(shap_predict, X_shap[:10])
#shap_vals = explainer.shap_values(X_shap [:50])
explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(X_tensor)


if isinstance(shap_values, list):
    shap_vals = shap_values[0]
else:
    shap_vals = shap_values


print("shap_vals shape:", shap_vals.shape)


feature_names = processor.df.columns.tolist()

num_outputs = shap_vals.shape[1]  # 12 in your case

for target_idx in range(num_outputs):
    target_name = trainer.present_targets[target_idx]
    shap_vals_target = shap_vals[:, target_idx, :]
    mean_abs_shap = np.mean(np.abs(shap_vals_target), axis=0)

    top_n = min(16, mean_abs_shap.shape[0])
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]

    print(f"\nTop {top_n} wichtigste Features für {target_name}:")
    for rank, feat_idx in enumerate(top_indices, start=1):
        feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feature {feat_idx}"
        value = mean_abs_shap[feat_idx]
        print(f"{rank}. {feat_name} (Feature {feat_idx}) shap = {value: .4f}")"""
