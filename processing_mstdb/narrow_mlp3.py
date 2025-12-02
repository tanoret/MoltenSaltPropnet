
"""
Train narrow3 MLP on MSTDB data with 3-fold cross-validation (80/20 splits),
NaN-masked multi-target regression, and save:
- Best model         → best_models/narrow_mlp/narrow_mlp_best.keras
- Best model metadata→ best_models/narrow_mlp/narrow_mlp_best_meta.json
- CV metrics / loss  → evaluate_modelperformance/narrow_mlp/cv_results.json
"""

import os
import sys
import json
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# 0. IMPORT YOUR PROCESSOR (DO NOT DEFINE IT HERE)
# ============================================================

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from processing_mstdb.processor import MSTDBProcessor


# ============================================================
# 1. ROBUST compute_composition (handles numeric Mol Frac)
# ============================================================

def robust_compute_composition(self, row, composition_type='elements'):
    """
    More robust version of compute_composition:
    - Mol Frac can be:
        * "Pure Salt"
        * "0.2-0.8"
        * "0.3" (string)
        * 0.3  (numeric)
        * NaN  (assume pure or equal fractions)
    """
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
    elif mol_frac is not None and not pd.isna(mol_frac):
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


# ============================================================
# 2. MASKED LOSS FOR MISSING TARGETS
# ============================================================

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


# ============================================================
# 3. MLP MODEL (with regularization & gradient clipping)
# ============================================================

def build_mlp(input_dim: int, n_targets: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="features")

    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(
        16,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(x)

    outputs = layers.Dense(n_targets, activation="linear", name="targets")(x)

    optimizer = keras.optimizers.AdamW(
        learning_rate=5e-4,
        weight_decay=1e-4,
    )

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=masked_mse)
    return model


# ============================================================
# 4. DATA PREPARATION (features, scaling, ALL target variables)
# ============================================================

ALL_TARGET_COLS = [
    "Melt(K)", "Boil(K)",
    "rho_a", "rho_b",
    "mu1_a", "mu1_b",
    "mu2_a", "mu2_b", "mu2_c",
    "k_a", "k_b",
    "cp_a", "cp_b",
]


def safe_target_stats(Y_train_orig: np.ndarray):
    """
    Compute per-column mean/std, handling the case where a column is all NaN.
    For all-NaN columns:
      mean = 0, std = 1
    so scaling is well-defined and those outputs simply don't contribute to loss.
    """
    n_targets = Y_train_orig.shape[1]
    means = np.zeros(n_targets, dtype=np.float32)
    stds = np.ones(n_targets, dtype=np.float32)

    for j in range(n_targets):
        col = Y_train_orig[:, j]
        mask = ~np.isnan(col)
        if mask.sum() == 0:
            # no labels at all for this target in train
            means[j] = 0.0
            stds[j] = 1.0
        else:
            m = col[mask].mean()
            s = col[mask].std()
            if s == 0:
                s = 1.0
            means[j] = m
            stds[j] = s

    return means, stds


def build_feature_matrix(processor: MSTDBProcessor) -> Dict[str, Any]:
    """
    Build full feature/target matrices (no splitting).
    """
    df = processor.df.copy()

    # Composition dict per row (elements)
    df["Composition"] = df.apply(
        lambda row: processor.compute_composition(row, composition_type="elements"),
        axis=1
    )

    # Element-fraction features
    elements = sorted(processor.predefined_elements)
    for el in elements:
        df[f"elem_{el}"] = df["Composition"].apply(lambda d: d.get(el, 0.0))

    # Extra simple features
    df["is_mixture"] = df["System"].apply(lambda s: 1.0 if "-" in str(s) else 0.0)
    df["n_elements"] = df["Composition"].apply(lambda d: float(len(d)))

    # Ensure all targets exist
    for col in ALL_TARGET_COLS:
        if col not in df.columns:
            df[col] = np.nan

    target_cols = list(ALL_TARGET_COLS)

    # Feature matrix (fill NaNs with 0)
    feat_cols = ["Mol Mass"] + [f"elem_{el}" for el in elements] + ["is_mixture", "n_elements"]
    X_df = df[feat_cols].astype("float32").fillna(0.0)
    X_all = X_df.values.astype("float32")

    # Targets (keep NaNs, they will be masked)
    Y_all = df[target_cols].astype("float32").values

    print("Full X_all shape:", X_all.shape)
    print("Full Y_all shape:", Y_all.shape)
    print("Targets used:", target_cols)

    return {
        "df": df,
        "feat_cols": feat_cols,
        "target_cols": target_cols,
        "elements": elements,
        "X_all": X_all,
        "Y_all": Y_all,
    }


def prepare_fold_data(
    feat_data: Dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Dict[str, Any]:
    """
    Given full feature matrix and indices, prepare scaled train/val for this fold.
    """
    X_all = feat_data["X_all"]
    Y_all = feat_data["Y_all"]

    X_train = X_all[train_idx]
    X_val = X_all[val_idx]
    Y_train_orig = Y_all[train_idx]
    Y_val_orig = Y_all[val_idx]

    # Scale features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)

    # Ensure no NaN/Inf after scaling
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale targets with safe stats (handles all-NaN columns)
    target_means, target_stds = safe_target_stats(Y_train_orig)
    Y_train_scaled = (Y_train_orig - target_means) / target_stds
    Y_val_scaled = (Y_val_orig - target_means) / target_stds

    print("Fold: X_train_scaled shape", X_train_scaled.shape,
          "NaN:", np.isnan(X_train_scaled).any(),
          "Inf:", np.isinf(X_train_scaled).any())
    print("Fold: Y_train_scaled shape", Y_train_scaled.shape,
          "NaN (OK, masked in loss):", np.isnan(Y_train_scaled).any(),
          "Inf:", np.isinf(Y_train_scaled).any())

    return {
        "X_train": X_train_scaled.astype("float32"),
        "X_val": X_val_scaled.astype("float32"),
        "Y_train": Y_train_scaled.astype("float32"),
        "Y_val": Y_val_scaled.astype("float32"),
        "Y_train_orig": Y_train_orig.astype("float32"),
        "Y_val_orig": Y_val_orig.astype("float32"),
        "target_means": target_means.astype("float32"),
        "target_stds": target_stds.astype("float32"),
        "scaler_X": scaler_X,
        "target_cols": feat_data["target_cols"],
        "feat_cols": feat_data["feat_cols"],
        "elements": feat_data["elements"],
        "df": feat_data["df"],
    }


# ============================================================
# 5. PER-TARGET METRICS (MSE & R²)
# ============================================================

def evaluate_per_target(model: keras.Model, data: Dict[str, Any]) -> Dict[str, Any]:
    X_train = data["X_train"]
    X_val = data["X_val"]
    Y_train_orig = data["Y_train_orig"]
    Y_val_orig = data["Y_val_orig"]
    means = data["target_means"]
    stds = data["target_stds"]
    target_cols = data["target_cols"]

    Y_train_pred_scaled = model.predict(X_train, verbose=0)
    Y_val_pred_scaled = model.predict(X_val, verbose=0)

    Y_train_pred = Y_train_pred_scaled * stds + means
    Y_val_pred = Y_val_pred_scaled * stds + means

    metrics = {"train": {}, "val": {}}

    for j, name in enumerate(target_cols):
        # TRAIN
        y_true_tr = Y_train_orig[:, j]
        y_pred_tr = Y_train_pred[:, j]
        mask_tr = (~np.isnan(y_true_tr)) & np.isfinite(y_pred_tr)

        if mask_tr.sum() > 1:
            mse_tr = mean_squared_error(y_true_tr[mask_tr], y_pred_tr[mask_tr])
            r2_tr = r2_score(y_true_tr[mask_tr], y_pred_tr[mask_tr])
        else:
            mse_tr = np.nan
            r2_tr = np.nan

        # VAL
        y_true_val = Y_val_orig[:, j]
        y_pred_val = Y_val_pred[:, j]
        mask_val = (~np.isnan(y_true_val)) & np.isfinite(y_pred_val)

        if mask_val.sum() > 1:
            mse_val = mean_squared_error(y_true_val[mask_val], y_pred_val[mask_val])
            r2_val = r2_score(y_true_val[mask_val], y_pred_val[mask_val])
        else:
            mse_val = np.nan
            r2_val = np.nan

        metrics["train"][name] = {"MSE": mse_tr, "R2": r2_tr}
        metrics["val"][name] = {"MSE": mse_val, "R2": r2_val}

    return metrics


# ============================================================
# 6. CROSS-VALIDATION TRAINING
# ============================================================

BEST_MODEL_DIR = os.path.join("best_models", "narrow_mlp3")
OUTDIR = os.path.join("evaluate_modelperformance", "narrow_mlp3")
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)


def run_cross_validation_narrow_mlp(
    processor: MSTDBProcessor,
    k_folds: int = 3,
    epochs: int = 400,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    3-fold cross-validation using repeated 80/20 splits (ShuffleSplit).
    Saves:
      - best model + metadata
      - cv_results.json (metrics + loss curves)
    """
    feat_data = build_feature_matrix(processor)
    n_samples = feat_data["X_all"].shape[0]

    splitter = ShuffleSplit(
        n_splits=k_folds,
        test_size=0.2,   # 80/20 splits, as requested
        random_state=42,
    )

    cv_results: Dict[str, Any] = {"folds": []}
    global_best_val = float("inf")
    global_best_fold = None

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(np.arange(n_samples))):
        print(f"\n================= FOLD {fold_idx + 1}/{k_folds} =================")
        print("Train size:", len(train_idx), "Val size:", len(val_idx))

        data = prepare_fold_data(feat_data, train_idx, val_idx)

        input_dim = data["X_train"].shape[1]
        n_targets = data["Y_train"].shape[1]

        model = build_mlp(input_dim, n_targets)
        model.summary()

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=20,
                restore_best_weights=True,
            )
        ]

        history = model.fit(
            data["X_train"],
            data["Y_train"],
            validation_data=(data["X_val"], data["Y_val"]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        metrics = evaluate_per_target(model, data)

        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        best_val_this_fold = float(np.min(val_loss))

        print(f"Fold {fold_idx} best val_loss = {best_val_this_fold:.6f}")

        fold_entry = {
            "fold": int(fold_idx),
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
            "best_val_loss": best_val_this_fold,
            "final_train_loss": float(loss[-1]),
            "final_val_loss": float(val_loss[-1]),
            "history": {
                "loss": [float(x) for x in loss],
                "val_loss": [float(x) for x in val_loss],
            },
            "metrics": {
                "train": {
                    k: {"MSE": float(v["MSE"]), "R2": float(v["R2"])}
                    for k, v in metrics["train"].items()
                },
                "val": {
                    k: {"MSE": float(v["MSE"]), "R2": float(v["R2"])}
                    for k, v in metrics["val"].items()
                },
            },
        }
        cv_results["folds"].append(fold_entry)

        # Track and save global best model
        if best_val_this_fold < global_best_val:
            global_best_val = best_val_this_fold
            global_best_fold = fold_idx

            best_model_path = os.path.join(BEST_MODEL_DIR, "narrow_mlp_best3.keras")
            print(f"New best model found on fold {fold_idx}, saving to {best_model_path}")
            model.save(best_model_path)

            meta = {
                "feat_cols": feat_data["feat_cols"],
                "target_cols": feat_data["target_cols"],
                "elements": feat_data["elements"],
                "target_means": data["target_means"].tolist(),
                "target_stds": data["target_stds"].tolist(),
                "scaler_X_mean": data["scaler_X"].mean_.tolist(),
                "scaler_X_scale": data["scaler_X"].scale_.tolist(),
            }
            meta_path = os.path.join(BEST_MODEL_DIR, "narrow_mlp_best_meta3.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=4)
            print(f"Best model metadata saved to {meta_path}")

    cv_results["best_fold"] = int(global_best_fold) if global_best_fold is not None else None
    cv_results["best_val_loss"] = float(global_best_val)

    cv_path = os.path.join(OUTDIR, "cv_results.json")
    with open(cv_path, "w") as f:
        json.dump(cv_results, f, indent=4)
    print(f"\nSaved CV results to: {cv_path}")
    print(f"Best fold: {global_best_fold} with val_loss={global_best_val:.6f}")

    return cv_results


# ============================================================
# 7. MAIN
# ============================================================

def main():
    # TODO: adjust CSV path to your actual location
    csv_path = "/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv"
    processor = MSTDBProcessor.from_csv(csv_path)
    processor.df.columns = processor.df.columns.str.strip()

    run_cross_validation_narrow_mlp(
        processor,
        k_folds=3,
        epochs=400,
        batch_size=32,
    )

    print("\n---------------- CROSS-VALIDATION COMPLETE ----------------")
    print("Best model + metadata in:", BEST_MODEL_DIR)
    print("CV metrics / loss curves in:", OUTDIR)


if __name__ == "__main__":
    main()
