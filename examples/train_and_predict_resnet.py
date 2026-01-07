import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import torch.nn as nn

# Local import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processing_saltdblean.processor import SALTDBLEANProcessor
from processing_saltdblean.resnet_trainer import ResNetMetaTrainer, TARGETS, DERIVED_PROPS, ELEMENT_FEATURE_COLS


# -------------------------------
# Load and preprocess data
# -------------------------------
processor = SALTDBLEANProcessor.from_csv(
    "/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz_with_ionic_polarizability.csv"
)
processor.df.columns = processor.df.columns.str.strip()

trainer = ResNetMetaTrainer(processor.df, TARGETS, DERIVED_PROPS, ELEMENT_FEATURE_COLS)
trainer.train_base()
trainer.train_meta()
trainer.debug_element_feature_usage({"Na": 0.5, "Cl": 0.5})

# numeric: which is closer to actual labels?
trainer.report_with_without_elements(split="train", metric="mae")
trainer.report_with_without_elements(split="test", metric="mae")

# you can also do mse/rmse/r2
trainer.report_with_without_elements(split="test", metric="rmse")
trainer.report_with_without_elements(split="test", metric="r2")

# -------------------------------
# QUICK SANITY TESTS (Element features really present and non-trivial)
# -------------------------------
print("\n==============================")
print("Sanity checks: feature shapes")
print("==============================")
print("feat_dim:", trainer.feat_dim)
print("X shape:", trainer.X.shape)
print("X_embedded shape:", trainer.X_embedded.shape)
print("n_elem_features:", len(ELEMENT_FEATURE_COLS))

# Check last 7 columns are not constant/zero (assumes ordering: [..., element_features])
try:
    last7_std = trainer.X[:, -len(ELEMENT_FEATURE_COLS):].std(axis=0)
    last7_mean = trainer.X[:, -len(ELEMENT_FEATURE_COLS):].mean(axis=0)
    print("\n==============================")
    print("Sanity checks: element features (last 7 columns of X)")
    print("==============================")
    print("Means:", last7_mean)
    print("Stds :", last7_std)
    print("Any non-trivial std? ->", bool(np.any(last7_std > 1e-8)))
except Exception as e:
    print("Could not compute last-7-column stats:", repr(e))

# Print the aggregated columns if present (trainer builds col__wmean)
if hasattr(trainer, "elem_feat_cols"):
    missing_cols = [c for c in trainer.elem_feat_cols if c not in trainer.df.columns]
    print("\n==============================")
    print("Sanity checks: aggregated element feature columns in df")
    print("==============================")
    if missing_cols:
        print("Missing aggregated columns (unexpected):", missing_cols)
    else:
        print(trainer.df[trainer.elem_feat_cols].head())


# -------------------------------
# OPTIONAL: quick dimension check inside predict()
# (This is a "black box" check: prediction should run without shape mismatch.)
# -------------------------------
print("\n==============================")
print("Sanity checks: predict() should run and return values")
print("==============================")
example_composition = {"Na": 0.5, "Cl": 0.5}
predicted_coeffs = trainer.predict(example_composition)

print("\nPredicted coefficients for 50-50 NaCl:")
for k, v in predicted_coeffs.items():
    print(f"{k}: {v:.4f}")

derived_props = trainer.derived(predicted_coeffs, 900)
print("\nDerived properties at 900K:")
for k, v in derived_props.items():
    print(f"{k}: {v:.4f}")


# -------------------------------
# Plot directory
# -------------------------------
plot_dir = "resnet_prediction_plots"
os.makedirs(plot_dir, exist_ok=True)

print("\nPlot directory:", plot_dir)
print("\nPlotting results...")


# -------------------------------
# Helper: model prediction for batches of FEATURES (NOT compositions)
# NOTE: X_input is expected to be trainer.X_embedded[...] or similar.
# -------------------------------
def predict_all(X_input: np.ndarray) -> np.ndarray:
    # base_nets is a ModuleDict; set eval on each net
    for net in trainer.base_nets.values():
        net.eval()
    trainer.meta.eval()

    with torch.no_grad():
        xb = torch.tensor(X_input, dtype=torch.float32, device=trainer.device)
        base_out = torch.stack([trainer.base_nets[p](xb) for p in trainer.present_targets], 1)
        pred_std = (base_out + trainer.meta(base_out)).cpu().numpy()
        return pred_std * trainer.σ + trainer.μ


# ---------------------------------------
# Coefficient Actual vs Predicted
# ---------------------------------------
for split_name, idx_set in zip(["train", "test"], [trainer.tr_idx, trainer.te_idx]):
    y_true = trainer.y_raw[idx_set]
    y_pred = predict_all(trainer.X_embedded[idx_set])

    for j, target in enumerate(trainer.present_targets):
        # Do not drop valid negatives; filter non-finite (and optionally near-zeros).
        mask = np.isfinite(y_true[:, j]) & (np.abs(y_true[:, j]) > 1e-12)

        if np.any(mask):
            plt.figure(figsize=(6, 6))
            plt.scatter(y_true[mask, j], y_pred[mask, j], alpha=0.7)
            plt.plot(
                [y_true[mask, j].min(), y_true[mask, j].max()],
                [y_true[mask, j].min(), y_true[mask, j].max()],
                "r--",
            )
            plt.title(f"{target} ({split_name} set)")
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.grid(True)
            plt.tight_layout()
            fname = f"actual_vs_predicted_coeff_{target}_{split_name}.png"
            plt.savefig(os.path.join(plot_dir, fname))
            plt.close()
            print(f"Saved: {fname}")


# ---------------------------------------
# Thermophysical Properties: Actual vs Predicted
# ---------------------------------------
print("\nPlotting actual vs predicted thermophysical properties...")

temperature = 900
properties_to_compare = ["rho", "muA", "muB", "k", "cp"]

for split_name, idx_set in zip(["train", "test"], [trainer.tr_idx, trainer.te_idx]):
    actual_vals_dict = {prop: [] for prop in properties_to_compare}
    predicted_vals_dict = {prop: [] for prop in properties_to_compare}

    for idx in idx_set:
        row = trainer.df.iloc[idx]

        # IMPORTANT: do not treat filled-0s as "actual" coefficients.
        # Use original availability mask for that row.
        mask_row = trainer.mask_all[idx]
        actual_coeffs = {}
        for k_i, col in enumerate(trainer.present_targets):
            if mask_row[k_i]:
                actual_coeffs[col] = float(row[col])

        actual_props = trainer.derived(actual_coeffs, temperature)

        pred_coeffs = dict(
            zip(
                trainer.present_targets,
                predict_all(trainer.X_embedded[[idx]])[0],
            )
        )
        pred_props = trainer.derived(pred_coeffs, temperature)

        for prop in properties_to_compare:
            a, p = actual_props.get(prop), pred_props.get(prop)
            if a is not None and p is not None and np.isfinite(a) and np.isfinite(p) and abs(a) > 1e-12:
                actual_vals_dict[prop].append(a)
                predicted_vals_dict[prop].append(p)

    for prop in properties_to_compare:
        if actual_vals_dict[prop]:
            plt.figure(figsize=(6, 6))
            plt.scatter(actual_vals_dict[prop], predicted_vals_dict[prop], alpha=0.7)
            plt.plot(
                [min(actual_vals_dict[prop]), max(actual_vals_dict[prop])],
                [min(actual_vals_dict[prop]), max(actual_vals_dict[prop])],
                "r--",
            )
            plt.title(f"{prop} at {temperature} K ({split_name} set)")
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.grid(True)
            plt.tight_layout()
            fname = f"actual_vs_predicted_property_{prop}_{split_name}.png"
            plt.savefig(os.path.join(plot_dir, fname))
            plt.close()
            print(f"Saved: {fname}")

print("\nAll plots saved in", plot_dir)


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
