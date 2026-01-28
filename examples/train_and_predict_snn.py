import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------------------------------
# Local imports
# ------------------------------------------------------------
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
    "NaCl_50_50_atoms": {"Na": 0.5, "Cl": 0.5},
}

# PCA settings
EMBEDDING_METHOD = "pca"
N_COMPONENTS = 32

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODELS_A_DIR, exist_ok=True)
os.makedirs(MODELS_B_DIR, exist_ok=True)


# ============================================================
# Helper: relative MSE %
# ============================================================
def rel_mse_pct(y_true, y_pred):
    if y_true.size == 0:
        return float("nan")
    mse = mean_squared_error(y_true, y_pred)
    denom = float(np.mean(y_true ** 2)) or 1e-12
    return 100.0 * mse / denom


# ============================================================
# Load dataset
# ============================================================
processor = SALTDBLEANProcessor.from_csv(DATA_PATH)
processor.df.columns = processor.df.columns.str.strip()
df = processor.df

# ============================================================
# Shared Splits
# ============================================================
idx_all = np.arange(len(df))
tr_idx, te_idx = train_test_split(idx_all, test_size=0.20, random_state=SEED)
tr_idx, va_idx = train_test_split(tr_idx, test_size=0.20, random_state=SEED)
splits = (tr_idx, va_idx, te_idx)


# ============================================================
# Build A/B Trainers
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

assert trainerA.present_targets == trainerB.present_targets, "Target mismatch between A and B!"


# ============================================================
# Train A / Train B
# ============================================================
print("\n==============================")
print("TRAIN A (SNN): WITHOUT element features")
print("==============================")
trainerA.train_base()

# NEW — clean meta printout
print("\n==============================")
print("Stage-2: Meta Training (A)")
print("==============================")
print("Meta optimizing targets:")
for t in trainerA.present_targets:
    print(f" • Meta learning for {t}")

trainerA.train_meta()
trainerA.evaluate("test")


print("\n==============================")
print("TRAIN B (SNN): WITH element features")
print("==============================")
trainerB.train_base()

print("\n==============================")
print("Stage-2: Meta Training (B)")
print("==============================")
print("Meta optimizing targets:")
for t in trainerB.present_targets:
    print(f" • Meta learning for {t}")

trainerB.train_meta()
trainerB.evaluate("test")


# ============================================================
# Prediction Helper
# ============================================================
def predict_all_embedded(trainer, X_embedded):
    for net in trainer.base_nets.values():
        net.eval()
    trainer.meta.eval()

    with torch.no_grad():
        xb = torch.tensor(X_embedded, dtype=torch.float32, device=trainer.device)
        base_out = torch.cat([trainer.base_nets[p](xb) for p in trainer.present_targets], dim=1)
        pred_std = (base_out + trainer.meta(base_out)).cpu().numpy()
        return pred_std * trainer.σ + trainer.μ


# ============================================================
# Element Filter
# ============================================================
def indices_with_element(trainer, element: str, split="test", min_frac=1e-12):
    split_map = {"train": trainer.tr_idx, "val": trainer.va_idx, "test": trainer.te_idx}
    idxs = split_map[split]
    frac = trainer.composition_df.loc[idxs, element].to_numpy(float)
    return idxs[frac > float(min_frac)]


# ============================================================
# Metrics Evaluation
# ============================================================
def evaluate_subset(trainer, idxs):
    if len(idxs) == 0:
        return {"n": 0, "per_target": {}, "avg_mse": np.nan, "avg_r2": np.nan}

    pred = predict_all_embedded(trainer, trainer.X_embedded[idxs])
    y_true = trainer.y_raw[idxs]
    mask = trainer.mask_all[idxs].astype(bool)

    per_target = {}
    mse_list = []
    r2_list = []

    for j, target in enumerate(trainer.present_targets):
        m = mask[:, j]
        if not np.any(m):
            continue

        yt = y_true[m, j]
        yp = pred[m, j]

        mse_pct = rel_mse_pct(yt, yp)
        r2_val = r2_score(yt, yp)

        per_target[target] = {
            "MSE_pct": float(mse_pct),
            "R2": float(r2_val),
            "N": int(m.sum())
        }

        mse_list.append(mse_pct)
        r2_list.append(r2_val)

    return {
        "n": len(idxs),
        "per_target": per_target,
        "avg_mse": float(np.mean(mse_list)),
        "avg_r2": float(np.mean(r2_list)),
    }


def write_metrics(path, title, metrics):
    with open(path, "a") as f:
        f.write("\n" + title + "\n")
        f.write("=" * 80 + "\n")
        f.write(f"N rows: {metrics['n']}\n")
        f.write(f"Avg MSE%: {metrics['avg_mse']:.6f}\n")
        f.write(f"Avg R2:   {metrics['avg_r2']:.6f}\n\n")
        f.write(f"{'Target':22s} | {'MSE%':>12s} | {'R2':>10s} | {'N':>6s}\n")
        f.write("-" * 80 + "\n")

        for t, d in metrics["per_target"].items():
            f.write(f"{t:22s} | {d['MSE_pct']:12.6f} | {d['R2']:10.6f} | {d['N']:6d}\n")


metrics_file = os.path.join(PLOT_DIR, "metrics_per_target_A_vs_B.txt")
with open(metrics_file, "w") as f:
    f.write("SNN A/B experiment metrics\n\n")

# Test set
write_metrics(metrics_file, "TEST SET — MODEL A (no element features)", evaluate_subset(trainerA, trainerA.te_idx))
write_metrics(metrics_file, "TEST SET — MODEL B (with element features)", evaluate_subset(trainerB, trainerB.te_idx))

# Element subsets
for el in ELEMENT_FILTERS:
    idxs = indices_with_element(trainerA, el, "test")
    write_metrics(metrics_file, f"ELEMENT {el} — MODEL A", evaluate_subset(trainerA, idxs))
    write_metrics(metrics_file, f"ELEMENT {el} — MODEL B", evaluate_subset(trainerB, idxs))


# ============================================================
# Plotting Functions (coefficients + derived)
# ============================================================
def plot_coeffs_A_vs_B(trA, trB, idxs, title, outdir, fname_prefix, max_targets=8):
    if len(idxs) == 0:
        print(f"[WARN] No rows for {title}. Skipping.")
        return

    y_true = trA.y_raw[idxs]
    mask = trA.mask_all[idxs]
    predA = predict_all_embedded(trA, trA.X_embedded[idxs])
    predB = predict_all_embedded(trB, trB.X_embedded[idxs])

    targets = trA.present_targets
    n = len(targets)

    for start in range(0, n, max_targets):
        chunk = targets[start:start+max_targets]
        rows = len(chunk)

        fig, axes = plt.subplots(rows, 2, figsize=(13, 4.0 * rows))

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

            # A
            mn = min(x.min(), yA.min())
            mx = max(x.max(), yA.max())
            axA.scatter(x, yA, alpha=0.7)
            axA.plot([mn, mx], [mn, mx], "k--")
            axA.set_title(f"{t} | WITHOUT elem-feats")

            # B
            mn = min(x.min(), yB.min())
            mx = max(x.max(), yB.max())
            axB.scatter(x, yB, alpha=0.7)
            axB.plot([mn, mx], [mn, mx], "k--")
            axB.set_title(f"{t} | WITH elem-feats")

        fig.suptitle(title)
        fig.tight_layout()
        fname = f"{fname_prefix}_coeff_A_vs_B_{start}.png"
        fig.savefig(os.path.join(outdir, fname), dpi=160, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", fname)


# ============================================================
# Run element-filter plots
# ============================================================
print("\nSaving plots...")

for el in ELEMENT_FILTERS:
    idxs = indices_with_element(trainerA, el, "test")
    title = f"Element {el} > 0 | N={len(idxs)}"
    prefix = f"{el}_test"

    plot_coeffs_A_vs_B(trainerA, trainerB, idxs, title, PLOT_DIR, prefix)


print("\nDone.")
print("Plots saved to:", PLOT_DIR)
print("Metrics saved to:", metrics_file)
print("Models A:", MODELS_A_DIR)
print("Models B:", MODELS_B_DIR)
