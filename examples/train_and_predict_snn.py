"""
train_and_predict_snn.py  —  end-to-end smoke-test for SNNMetaTrainer
--------------------------------------------------------------------
✓ trains base SNNs
✓ trains meta network with physics regularisation
✓ predicts coefficients & derived props for 50-50 NaCl
✓ makes parity plots for train / test splits (coeffs + thermo-props)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# ────────────────────────────────────────────────────────────────
#  Make local package importable
# ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processing_saltdblean.processor    import SALTDBLEANProcessor
from processing_saltdblean.snn_trainer  import SNNMetaTrainer, TARGETS, DERIVED_PROPS

# ────────────────────────────────────────────────────────────────
#  1. Load & preprocess database
# ────────────────────────────────────────────────────────────────
processor = SALTDBLEANProcessor.from_csv("../data/saltdblean_processed.csv")
processor.df.columns = processor.df.columns.str.strip()   # remove stray spaces

# ────────────────────────────────────────────────────────────────
#  2. Train SNN + meta + physics regularisation
# ────────────────────────────────────────────────────────────────
trainer = SNNMetaTrainer(processor.df, TARGETS, DERIVED_PROPS)
trainer.train_base()
trainer.train_meta()

# switch all nets to evaluation mode once
for net in trainer.base_nets.values():
    net.eval()
trainer.meta.eval()

# ────────────────────────────────────────────────────────────────
#  3. Quick demo prediction
# ────────────────────────────────────────────────────────────────
example = {"Na": 0.5, "Cl": 0.5}
coeff = trainer.predict(example)

print("\nPredicted coefficients for 50-50 NaCl:")
for k, v in coeff.items():
    print(f"{k:7s}: {v:11.4f}")

print("\nDerived properties at 900 K:")
derived = trainer.derived(coeff, 900)
for k, v in derived.items():
    print(f"{k:4s}: {v:11.4f}")

# ────────────────────────────────────────────────────────────────
#  4. Helper: batch prediction (physical units)
# ────────────────────────────────────────────────────────────────
def predict_all(X_input: np.ndarray) -> np.ndarray:
    xb = torch.tensor(X_input, dtype=torch.float32, device=trainer.device)
    with torch.no_grad():
        # concatenate → shape (B,14)  (stack would give (B,14,1))
        base_std = torch.cat([trainer.base_nets[p](xb)
                              for p in trainer.present_targets], dim=1)
        pred_std = base_std + trainer.meta(base_std)
    return pred_std.cpu().numpy() * trainer.σ + trainer.μ   # back to physical units

# ────────────────────────────────────────────────────────────────
#  5. Parity plots — coefficients
# ────────────────────────────────────────────────────────────────
print("\nPlotting results …")
plot_dir = "snn_prediction_plots"
os.makedirs(plot_dir, exist_ok=True)

for split, idx_set in zip(["train", "test"], [trainer.tr_idx, trainer.te_idx]):

    y_true = trainer.y_raw[idx_set]
    y_pred = predict_all(trainer.X[idx_set])

    for j, tgt in enumerate(trainer.present_targets):
        mask = y_true[:, j] > 1e-10
        if np.any(mask):
            plt.figure(figsize=(6, 6))
            plt.scatter(y_true[mask, j], y_pred[mask, j], alpha=0.7)
            lims = [y_true[mask, j].min(), y_true[mask, j].max()]
            plt.plot(lims, lims, "r--")
            plt.title(f"{tgt} ({split} set)")
            plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.grid(True)
            plt.tight_layout()
            fname = f"actual_vs_predicted_coeff_{tgt}_{split}.png"
            plt.savefig(os.path.join(plot_dir, fname))
            plt.close()
            print(f"Saved: {fname}")

# ────────────────────────────────────────────────────────────────
#  6. Parity plots — derived thermo-physical properties
# ────────────────────────────────────────────────────────────────
print("\nPlotting actual vs. predicted thermo-physical properties …")

T_plot  = 900                                     # K
props   = ["rho", "muA", "muB", "k", "cp"]

for split, idx_set in zip(["train", "test"], [trainer.tr_idx, trainer.te_idx]):

    actual, pred = {p: [] for p in props}, {p: [] for p in props}

    for idx in idx_set:
        # ground-truth coefficients → properties
        row_coeff = {c: trainer.df.iloc[idx][c] for c in trainer.present_targets}
        a_props   = trainer.derived(row_coeff, T_plot)

        # model coefficients → properties
        m_coeff   = dict(zip(trainer.present_targets,
                             predict_all(trainer.X[[idx]])[0]))
        p_props   = trainer.derived(m_coeff, T_plot)

        for p in props:
            a = a_props.get(p);  pr = p_props.get(p)
            if a is not None and pr is not None and a > 1e-6:
                actual[p].append(a); pred[p].append(pr)

    for p in props:
        if actual[p]:
            plt.figure(figsize=(6, 6))
            plt.scatter(actual[p], pred[p], alpha=0.7)
            lims = [min(actual[p]), max(actual[p])]
            plt.plot(lims, lims, "r--")
            plt.title(f"{p} at {T_plot} K ({split} set)")
            plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.grid(True)
            plt.tight_layout()
            fname = f"actual_vs_predicted_property_{p}_{split}.png"
            plt.savefig(os.path.join(plot_dir, fname))
            plt.close()
            print(f"Saved: {fname}")
        else:
            print(f"Skipped {p} ({split}) — not enough data")

print("\nAll plots saved in", plot_dir)
