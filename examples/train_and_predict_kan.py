import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Ensure local module path is visible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processing_saltdblean.processor import SALTDBLEANProcessor
from processing_saltdblean.kan_trainer import KANMetaTrainer, TARGETS, DERIVED_PROPS

# Step 1: Load and preprocess the data
processor = SALTDBLEANProcessor.from_csv('../data/saltdblean_processed.csv')
processor.df.columns = processor.df.columns.str.strip()  # clean column names

# Step 2: Train KAN+Meta+Physics
trainer = KANMetaTrainer(processor.df, TARGETS, DERIVED_PROPS)
trainer.train_base()
trainer.train_meta()

# Step 3: Predict for an example
example_composition = {'Na': 0.5, 'Cl': 0.5}
predicted_coeffs = trainer.predict(example_composition)
print("\nPredicted coefficients for 50-50 NaCl:")
for k, v in predicted_coeffs.items():
    print(f"{k}: {v:.4f}")

# Step 4: Compute derived properties at 900K
derived_props = trainer.derived(predicted_coeffs, 900)
print("\nDerived properties at 900K:")
for k, v in derived_props.items():
    print(f"{k}: {v:.4f}")

# Step 5: Plotting
print("\nPlotting results...")
plot_dir = "kan_prediction_plots"
os.makedirs(plot_dir, exist_ok=True)

# Helper to predict batches
def predict_all(X_input):
    import torch
    trainer.base_nets.eval()
    trainer.meta.eval()
    with torch.no_grad():
        xb = torch.tensor(X_input, dtype=torch.float32, device=trainer.device)
        base_out = torch.stack([trainer.base_nets[p](xb) for p in trainer.present_targets], 1)
        pred = (base_out + trainer.meta(base_out)).cpu().numpy()
        return pred * trainer.σ + trainer.μ

# Step 6: Actual vs Predicted - coefficients
for split_name, idx_set in zip(["train", "test"], [trainer.tr_idx, trainer.te_idx]):
    y_true = trainer.y_raw[idx_set]
    y_pred = predict_all(trainer.X[idx_set])

    for j, target in enumerate(trainer.present_targets):
        mask = y_true[:, j] > 1e-10
        if np.any(mask):
            plt.figure(figsize=(6, 6))
            plt.scatter(y_true[mask, j], y_pred[mask, j], alpha=0.7)
            plt.plot([y_true[mask, j].min(), y_true[mask, j].max()],
                    [y_true[mask, j].min(), y_true[mask, j].max()], 'r--')
            plt.title(f"{target} ({split_name} set)")
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.grid(True)
            plt.tight_layout()
            fname = f"actual_vs_predicted_coeff_{target}_{split_name}.png"
            plt.savefig(os.path.join(plot_dir, fname))
            plt.close()
            print(f"Saved: {fname}")

# Step 7: Actual vs Predicted - thermophysical properties
def compute_actual_properties_from_coeffs(row, temperature):
    coeff = {col: row.get(col, 0.0) for col in trainer.present_targets}
    return trainer.derived(coeff, temperature)

print("\nPlotting actual vs. predicted thermophysical properties...")

temperature = 900  # Kelvin
properties_to_compare = ["rho", "muA", "muB", "k", "cp"]

for split_name, idx_set in zip(["train", "test"], [trainer.tr_idx, trainer.te_idx]):
    actual_vals_dict = {prop: [] for prop in properties_to_compare}
    predicted_vals_dict = {prop: [] for prop in properties_to_compare}

    for idx in idx_set:
        row = trainer.df.iloc[idx]
        actual_coeffs = {col: row.get(col, 0.0) for col in trainer.present_targets}
        actual_props = trainer.derived(actual_coeffs, temperature)

        pred_coeffs = dict(zip(trainer.present_targets, predict_all(trainer.X[[idx]])[0]))
        pred_props = trainer.derived(pred_coeffs, temperature)

        for prop in properties_to_compare:
            a = actual_props.get(prop, None)
            p = pred_props.get(prop, None)
            if a is not None and p is not None and a > 1e-6:
                actual_vals_dict[prop].append(a)
                predicted_vals_dict[prop].append(p)

    for prop in properties_to_compare:
        if actual_vals_dict[prop] and predicted_vals_dict[prop]:
            plt.figure(figsize=(6, 6))
            plt.scatter(actual_vals_dict[prop], predicted_vals_dict[prop], alpha=0.7)
            plt.plot([min(actual_vals_dict[prop]), max(actual_vals_dict[prop])],
                     [min(actual_vals_dict[prop]), max(actual_vals_dict[prop])], 'r--')
            plt.title(f"{prop} at {temperature} K ({split_name} set)")
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.grid(True)
            plt.tight_layout()
            fname = f"actual_vs_predicted_property_{prop}_{split_name}.png"
            plt.savefig(os.path.join(plot_dir, fname))
            plt.close()
            print(f"Saved: {fname}")
        else:
            print(f"Skipped {prop} ({split_name}) — no valid data.")

print("\nAll plots saved in", plot_dir)
