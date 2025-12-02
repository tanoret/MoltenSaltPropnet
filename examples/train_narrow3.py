#!/usr/bin/env python
"""
Visualizations for narrow MLP CV run.

Loads:
    evaluate_modelperformance/narrow_mlp/cv_results.json
    best_models/narrow_mlp/narrow_mlp_best.keras
    best_models/narrow_mlp/narrow_mlp_best_meta.json

Produces:
    - loss_curve_best_fold.png       : train vs val loss (best fold)
    - cv_r2_boxplot.png              : per-target CV R² stability (val across folds)
    - true_vs_pred_best_fold.png     : per-target true vs predicted (best fold val set)
"""

import os
import json
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from processing_mstdb.processor import MSTDBProcessor

import tensorflow as tf
from tensorflow import keras

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

OUTDIR = os.path.join("evaluate_modelperformance", "narrow_mlp3")
BEST_MODEL_DIR = os.path.join("best_models", "narrow_mlp3")

os.makedirs(OUTDIR, exist_ok=True)


# ------------------------------------------------------------------
# Same masked loss as in training (for loading the model)
# ------------------------------------------------------------------

def masked_mse(y_true, y_pred):
    """
    y_true, y_pred: (batch, n_targets)
    - y_true may contain NaN for missing labels.
    - We compute MSE only where y_true is not NaN.
    """
    mask = tf.math.logical_not(tf.math.is_nan(y_true))

    y_true_clean = tf.where(mask, y_true, tf.zeros_like(y_true))
    y_pred_clean = tf.where(mask, y_pred, tf.zeros_like(y_pred))

    sq_err = tf.square(y_true_clean - y_pred_clean)
    sq_err_masked = tf.where(mask, sq_err, tf.zeros_like(sq_err))

    sq_sum = tf.reduce_sum(sq_err_masked)
    n_valid = tf.reduce_sum(tf.cast(mask, tf.float32))
    n_valid = tf.maximum(n_valid, 1.0)
    return sq_sum / n_valid


# ------------------------------------------------------------------
# Robust compute_composition (same as in training)
# ------------------------------------------------------------------

def robust_compute_composition(self, row, composition_type='elements'):
    system = row['System']
    mol_frac = row['Mol Frac']
    compounds = [c.strip() for c in str(system).split('-')]

    fractions = None

    # Case 1: Mol Frac is a string
    if isinstance(mol_frac, str):
        s = mol_frac.strip()

        if s.lower() == 'pure salt':
            if len(compounds) != 1:
                raise ValueError(f"Pure Salt should have only one compound, got: {compounds}")
            fractions = [1.0]

        elif '-' in s:
            parts = [p.strip() for p in s.split('-') if p.strip() != ""]
            fractions = [float(p) for p in parts]

        else:
            # single numeric as string, e.g. "0.3"
            try:
                val = float(s)
            except ValueError:
                raise ValueError(f"Cannot interpret Mol Frac value: {mol_frac!r}")

            if len(compounds) == 1:
                fractions = [val]
            elif len(compounds) == 2:
                fractions = [val, 1.0 - val]
            else:
                raise ValueError(
                    f"Single numeric Mol Frac '{mol_frac}' with {len(compounds)} compounds is ambiguous."
                )

    # Case 2: Mol Frac is numeric (int/float, not NaN)
    elif mol_frac is not None and not np.isnan(mol_frac):
        val = float(mol_frac)
        if len(compounds) == 1:
            fractions = [val]
        elif len(compounds) == 2:
            fractions = [val, 1.0 - val]
        else:
            raise ValueError(
                f"Numeric Mol Frac {mol_frac} with {len(compounds)} compounds is ambiguous."
            )

    # Case 3: Mol Frac missing (NaN) → assume pure or equal fractions
    if fractions is None:
        if len(compounds) == 1:
            fractions = [1.0]
        else:
            fractions = [1.0 / len(compounds)] * len(compounds)

    if len(fractions) != len(compounds):
        raise ValueError(
            f"Number of fractions ({len(fractions)}) does not match number of compounds ({len(compounds)}): "
            f"Mol Frac={mol_frac!r}, System={system!r}"
        )

    # Compound-level composition
    compound_dict = {compound: frac for compound, frac in zip(compounds, fractions)}

    # Element-level composition
    total_composition = {}
    for compound, frac in zip(compounds, fractions):
        parsed_elements = self.parse_compound(compound)
        for element, count in parsed_elements.items():
            total_composition[element] = total_composition.get(element, 0.0) + frac * count

    total_sum = float(sum(total_composition.values()))
    if total_sum > 0:
        element_dict = {el: cnt / total_sum for el, cnt in total_composition.items()}
    else:
        element_dict = {}

    if composition_type == 'elements':
        return element_dict
    elif composition_type == 'compounds':
        return compound_dict
    elif composition_type == 'both':
        merged = dict(element_dict)
        merged.update(compound_dict)
        return merged
    else:
        raise ValueError("Invalid composition_type")


# Monkey-patch into MSTDBProcessor
MSTDBProcessor.compute_composition = robust_compute_composition


# ------------------------------------------------------------------
# Helpers for loading CV results & saving plots
# ------------------------------------------------------------------

def _load_cv_results() -> Dict:
    cv_path = os.path.join(OUTDIR, "cv_results.json")
    if not os.path.exists(cv_path):
        raise FileNotFoundError(
            f"cv_results.json not found at {cv_path}. "
            f"Run narrow_mlp_train.py first."
        )
    with open(cv_path, "r") as f:
        return json.load(f)


def _save_plot(filename: str):
    plt.tight_layout()
    path = os.path.join(OUTDIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


# ------------------------------------------------------------------
# 1) Train vs Val loss for best fold
# ------------------------------------------------------------------

def plot_best_fold_loss_curves(cv_results: Dict):
    best_fold = cv_results.get("best_fold", 0)
    folds = cv_results.get("folds", [])
    if not folds:
        print("No folds in cv_results; skipping loss curve.")
        return

    # Find the entry with that fold index
    fold_entry = None
    for f in folds:
        if int(f.get("fold", -1)) == int(best_fold):
            fold_entry = f
            break

    if fold_entry is None:
        print(f"Best fold {best_fold} not found in cv_results; using fold 0 instead.")
        fold_entry = folds[0]

    hist = fold_entry.get("history", {})
    loss = hist.get("loss", [])
    val_loss = hist.get("val_loss", [])

    if not loss or not val_loss:
        print("No loss history in best fold; skipping loss curve.")
        return

    epochs = np.arange(1, len(loss) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Masked MSE loss")
    plt.title(f"Train vs Val loss (best fold = {best_fold})")
    plt.legend()
    _save_plot("loss_curve_best_fold.png")


# ------------------------------------------------------------------
# 2) CV R² boxplot per target (val R² across folds)
# ------------------------------------------------------------------

def plot_cv_r2_boxplot(cv_results: Dict):
    folds = cv_results.get("folds", [])
    if not folds:
        print("No folds in cv_results; skipping CV R² boxplot.")
        return

    r2_per_target: Dict[str, List[float]] = {}

    for f in folds:
        metrics_val = f.get("metrics", {}).get("val", {})
        for t, m in metrics_val.items():
            r2 = m.get("R2", float("nan"))
            if np.isfinite(r2):
                r2_per_target.setdefault(t, []).append(r2)

    if not r2_per_target:
        print("No per-target val R² found; skipping CV R² boxplot.")
        return

    targets = sorted(r2_per_target.keys())
    data = [r2_per_target[t] for t in targets]

    plt.figure(figsize=(12, 5))
    plt.boxplot(data, labels=targets, showmeans=True)
    plt.axhline(0, color="black")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("R² (val across folds)")
    plt.title("Cross-validation R² stability per target (narrow MLP)")
    _save_plot("cv_r2_boxplot.png")


# ------------------------------------------------------------------
# 3) True vs predicted (best fold, validation set, per target)
# ------------------------------------------------------------------

def _build_full_feature_target_matrix(processor: MSTDBProcessor, feat_cols_meta, elements_meta, target_cols_meta):
    """
    Rebuild the same feature/target matrices as in training,
    using feat_cols from metadata to ensure identical ordering.
    """
    df = processor.df.copy()

    # Composition dict per row (elements)
    df["Composition"] = df.apply(
        lambda row: processor.compute_composition(row, composition_type="elements"),
        axis=1
    )

    # Element-fraction features (use the same element list as meta)
    for el in elements_meta:
        df[f"elem_{el}"] = df["Composition"].apply(lambda d: d.get(el, 0.0))

    # Extra simple features
    df["is_mixture"] = df["System"].apply(lambda s: 1.0 if "-" in str(s) else 0.0)
    df["n_elements"] = df["Composition"].apply(lambda d: float(len(d)))

    # Ensure targets exist
    for col in target_cols_meta:
        if col not in df.columns:
            df[col] = np.nan

    # Features: use exact order from meta
    X_df = df[feat_cols_meta].astype("float32").fillna(0.0)
    X_all = X_df.values.astype("float32")

    # Targets: same order as meta
    Y_all = df[target_cols_meta].astype("float32").values

    return df, X_all, Y_all


def plot_true_vs_pred_best_fold(cv_results: Dict, csv_path: str):
    """
    Load best model + metadata, rebuild data, recreate best fold split,
    and plot per-target true vs predicted on that validation split.
    """
    best_fold = cv_results.get("best_fold", 0)
    if best_fold is None:
        print("best_fold is None in cv_results; skipping true vs pred plot.")
        return

    # Load meta & model
    meta_path = os.path.join(BEST_MODEL_DIR, "narrow_mlp_best_meta3.json")
    model_path = os.path.join(BEST_MODEL_DIR, "narrow_mlp_best3.keras")

    if not os.path.exists(meta_path) or not os.path.exists(model_path):
        print("Best model or metadata not found; run training script first.")
        return

    with open(meta_path, "r") as f:
        meta = json.load(f)

    feat_cols = meta["feat_cols"]
    target_cols = meta["target_cols"]
    elements = meta["elements"]
    target_means = np.array(meta["target_means"], dtype=np.float32)
    target_stds = np.array(meta["target_stds"], dtype=np.float32)
    scaler_mean = np.array(meta["scaler_X_mean"], dtype=np.float32)
    scaler_scale = np.array(meta["scaler_X_scale"], dtype=np.float32)

    model = keras.models.load_model(
        model_path,
        custom_objects={"masked_mse": masked_mse},
    )

    # Load data and rebuild features
    processor = MSTDBProcessor.from_csv(csv_path)
    processor.df.columns = processor.df.columns.str.strip()

    df, X_all, Y_all = _build_full_feature_target_matrix(
        processor,
        feat_cols_meta=feat_cols,
        elements_meta=elements,
        target_cols_meta=target_cols,
    )

    # Recreate the same CV splits (ShuffleSplit) as in training
    from sklearn.model_selection import ShuffleSplit

    n_splits = cv_results.get("n_splits", 3)
    test_size = cv_results.get("test_size", 0.2)
    random_state = cv_results.get("random_state", 42)

    splitter = ShuffleSplit(
        n_splits=int(n_splits),
        test_size=float(test_size),
        random_state=int(random_state),
    )

    n_samples = X_all.shape[0]
    best_train_idx = None
    best_val_idx = None

    for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(np.arange(n_samples))):
        if fold_idx == int(best_fold):
            best_train_idx = tr_idx
            best_val_idx = va_idx
            break

    if best_val_idx is None:
        print("Could not reproduce best fold split; skipping true vs pred plot.")
        return

    # Validation subset for the best fold
    X_val = X_all[best_val_idx]
    Y_val_orig = Y_all[best_val_idx]

    # Apply the saved scaler and target un-scaling
    X_val_scaled = (X_val - scaler_mean) / scaler_scale
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    Y_pred_scaled = model.predict(X_val_scaled, verbose=0)
    Y_pred = Y_pred_scaled * target_stds + target_means

    # Per-target scatter plots (true vs pred)
    n_targets = len(target_cols)
    n_cols = 3
    n_rows = int(np.ceil(n_targets / n_cols))

    plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    plot_idx = 1

    for j, name in enumerate(target_cols):
        y_true = Y_val_orig[:, j]
        y_pred = Y_pred[:, j]

        mask = (~np.isnan(y_true)) & np.isfinite(y_pred)
        n_j = int(mask.sum())
        if n_j < 5:  # skip if too few points
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())

        plt.subplot(n_rows, n_cols, plot_idx)
        plt.scatter(yt, yp, alpha=0.5, s=10)
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        plt.title(name)
        plt.xlabel("True")
        plt.ylabel("Pred")

        plot_idx += 1

    plt.suptitle(f"True vs predicted (best fold={best_fold}, val set)", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save_plot("true_vs_pred_best_fold.png")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    cv_results = _load_cv_results()
    plot_best_fold_loss_curves(cv_results)
    plot_cv_r2_boxplot(cv_results)

    # Same CSV path you use in training:
    csv_path = "/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv"
    plot_true_vs_pred_best_fold(cv_results, csv_path)

    print("\nAll plots saved under:", OUTDIR)


if __name__ == "__main__":
    main()
