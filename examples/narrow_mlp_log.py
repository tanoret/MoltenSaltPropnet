import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
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

    optimizer = keras.optimizers.Adam(
        learning_rate=5e-4,
        clipnorm=1.0,
    )

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=masked_mse)
    return model


# ============================================================
# 4. TARGET LIST + TRANSFORMS
# ============================================================

ALL_TARGET_COLS = [
    "Melt(K)", "Boil(K)",
    "rho_a", "rho_b",
    "mu1_a", "mu1_b",
    "mu2_a", "mu2_b", "mu2_c",
    "k_a", "k_b",
    "cp_a", "cp_b",
]


def transform_targets(Y: np.ndarray, target_cols):
    """
    Apply per-target transforms for training:
      - Boil(K)  -> log(Boil)
      - mu1_a    -> log10(mu1_a)
    Others stay unchanged.
    Invalid values (<=0 for logs) become NaN (so they get masked).
    """
    Y_model = Y.copy().astype(np.float32)

    for j, name in enumerate(target_cols):
        col = Y_model[:, j]

        if name == "Boil(K)":
            # only positive temperatures make sense
            mask = col > 0
            col[~mask] = np.nan
            col[mask] = np.log(col[mask])
            Y_model[:, j] = col

        elif name == "mu1_a":
            # mu1_a should be positive (prefactor); if not, treat as missing
            mask = col > 0
            col[~mask] = np.nan
            col[mask] = np.log10(col[mask])
            Y_model[:, j] = col

        # else: leave as is

    return Y_model


def inverse_transform_targets(Y_model_pred: np.ndarray, target_cols):
    """
    Inverse of transform_targets:
      - exp for Boil(K)
      - 10**x for mu1_a
    """
    Y_phys = Y_model_pred.copy().astype(np.float32)

    for j, name in enumerate(target_cols):
        col = Y_phys[:, j]

        if name == "Boil(K)":
            col = np.exp(col)
            Y_phys[:, j] = col

        elif name == "mu1_a":
            col = np.power(10.0, col)
            Y_phys[:, j] = col

        # else: unchanged

    return Y_phys


def safe_target_stats(Y_train_model: np.ndarray):
    """
    Compute per-column mean/std on the TRANSFORMED targets (model space).
    Handles all-NaN columns:
      mean = 0, std = 1 so scaling is defined.
    """
    n_targets = Y_train_model.shape[1]
    means = np.zeros(n_targets, dtype=np.float32)
    stds  = np.ones(n_targets, dtype=np.float32)

    for j in range(n_targets):
        col = Y_train_model[:, j]
        mask = ~np.isnan(col)
        if mask.sum() == 0:
            means[j] = 0.0
            stds[j]  = 1.0
        else:
            m = col[mask].mean()
            s = col[mask].std()
            if s == 0:
                s = 1.0
            means[j] = m
            stds[j]  = s

    return means, stds


# ============================================================
# 5. DATA PREPARATION
# ============================================================

def prepare_dataset(processor: MSTDBProcessor):
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
    X = X_df.values

    # Targets in physical space
    Y_phys = df[target_cols].astype("float32").values

    # Train/val split on physical targets
    X_train, X_val, Y_train_phys, Y_val_phys = train_test_split(
        X, Y_phys, test_size=0.2, random_state=42
    )

    # Transform targets to model space for training
    Y_train_model = transform_targets(Y_train_phys, target_cols)
    Y_val_model   = transform_targets(Y_val_phys, target_cols)

    # Scale features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled   = scaler_X.transform(X_val)

    # Ensure no NaN/Inf after scaling
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_scaled   = np.nan_to_num(X_val_scaled,   nan=0.0, posinf=0.0, neginf=0.0)

    # Scale targets in model space
    target_means, target_stds = safe_target_stats(Y_train_model)
    Y_train_scaled = (Y_train_model - target_means) / target_stds
    Y_val_scaled   = (Y_val_model   - target_means) / target_stds

    print("X_train_scaled: shape", X_train_scaled.shape,
          "has NaN:", np.isnan(X_train_scaled).any(),
          "has inf:", np.isinf(X_train_scaled).any())
    print("Y_train_scaled: shape", Y_train_scaled.shape,
          "has NaN (OK, masked in loss):", np.isnan(Y_train_scaled).any(),
          "has inf:", np.isinf(Y_train_scaled).any())
    print("Targets used:", target_cols)

    return {
        "df": df,
        "feat_cols": feat_cols,
        "target_cols": target_cols,
        "elements": elements,

        # training inputs & targets (scaled, model space)
        "X_train": X_train_scaled.astype("float32"),
        "X_val":   X_val_scaled.astype("float32"),
        "Y_train": Y_train_scaled.astype("float32"),
        "Y_val":   Y_val_scaled.astype("float32"),

        # original physical targets (for metrics)
        "Y_train_phys": Y_train_phys.astype("float32"),
        "Y_val_phys":   Y_val_phys.astype("float32"),

        # scaling in model space
        "scaler_X": scaler_X,
        "target_means": target_means.astype("float32"),
        "target_stds":  target_stds.astype("float32"),
    }


# ============================================================
# 6. PER-TARGET METRICS (MSE & R² in PHYSICAL space)
# ============================================================

def evaluate_per_target(model: keras.Model, data: dict):
    X_train = data["X_train"]
    X_val   = data["X_val"]
    Y_train_phys = data["Y_train_phys"]
    Y_val_phys   = data["Y_val_phys"]
    means = data["target_means"]
    stds  = data["target_stds"]
    target_cols = data["target_cols"]

    # Predictions in model space (scaled)
    Y_train_pred_scaled = model.predict(X_train)
    Y_val_pred_scaled   = model.predict(X_val)

    # Unscale to model space
    Y_train_pred_model = Y_train_pred_scaled * stds + means
    Y_val_pred_model   = Y_val_pred_scaled   * stds + means

    # Inverse transform to physical space
    Y_train_pred_phys = inverse_transform_targets(Y_train_pred_model, target_cols)
    Y_val_pred_phys   = inverse_transform_targets(Y_val_pred_model,   target_cols)

    metrics = {"train": {}, "val": {}}

    for j, name in enumerate(target_cols):
        # TRAIN
        y_true_tr = Y_train_phys[:, j]
        y_pred_tr = Y_train_pred_phys[:, j]
        mask_tr = (~np.isnan(y_true_tr)) & np.isfinite(y_pred_tr)

        if mask_tr.sum() > 1:
            mse_tr = mean_squared_error(y_true_tr[mask_tr], y_pred_tr[mask_tr])
            r2_tr  = r2_score(y_true_tr[mask_tr], y_pred_tr[mask_tr])
        else:
            mse_tr = np.nan
            r2_tr  = np.nan

        # VAL
        y_true_val = Y_val_phys[:, j]
        y_pred_val = Y_val_pred_phys[:, j]
        mask_val = (~np.isnan(y_true_val)) & np.isfinite(y_pred_val)

        if mask_val.sum() > 1:
            mse_val = mean_squared_error(y_true_val[mask_val], y_pred_val[mask_val])
            r2_val  = r2_score(y_true_val[mask_val], y_pred_val[mask_val])
        else:
            mse_val = np.nan
            r2_val  = np.nan

        metrics["train"][name] = {"MSE": mse_tr, "R2": r2_tr}
        metrics["val"][name]   = {"MSE": mse_val, "R2": r2_val}

    return metrics


# ============================================================
# 7. HELPERS: PREDICT COEFFS + PROPERTIES AT T
# ============================================================

def predict_coefficients_for_all(trained: dict):
    """
    Predict ALL target coefficients for all rows in trained["df"].
    Returns list of dicts, one per row: {target_name: value} in PHYSICAL space.
    """
    df = trained["df"]
    feat_cols = trained["feat_cols"]
    target_cols = trained["target_cols"]
    scaler_X = trained["scaler_X"]
    means = trained["target_means"]
    stds  = trained["target_stds"]
    model = trained["model"]

    X_all = df[feat_cols].astype("float32").fillna(0.0).values
    X_all_scaled = scaler_X.transform(X_all)
    X_all_scaled = np.nan_to_num(X_all_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    Y_pred_scaled = model.predict(X_all_scaled)
    Y_pred_model  = Y_pred_scaled * stds + means
    Y_pred_phys   = inverse_transform_targets(Y_pred_model, target_cols)

    coeff_list = []
    for row_vals in Y_pred_phys:
        d = {name: float(v) for name, v in zip(target_cols, row_vals)}
        coeff_list.append(d)
    return coeff_list


def predict_properties_at_T(processor: MSTDBProcessor, trained: dict, T: float):
    """
    For each row in processor.df:
      - take ML predictions for all targets in ALL_TARGET_COLS (physical space)
      - add cp_c, cp_d from data row if present (or 0)
      - compute physical properties at temperature T via processor.compute_properties(...)
    Returns list of dicts (properties per row).
    """
    df = trained["df"].reset_index(drop=True)
    ml_coeffs_list = predict_coefficients_for_all(trained)

    props_list = []
    for idx, row in df.iterrows():
        coeffs = {}

        # start with all predicted coefficients
        coeffs.update(ml_coeffs_list[idx])

        # add cp_c, cp_d from data if present (else 0.0)
        for key in ["cp_c", "cp_d"]:
            if key in row.index and pd.notna(row[key]):
                coeffs[key] = float(row[key])
            else:
                coeffs[key] = coeffs.get(key, 0.0)

        props = processor.compute_properties(coeffs, T)
        props_list.append(props)

    return props_list


# ============================================================
# 8. TRAINING ROUTINE
# ============================================================

def train_model(processor: MSTDBProcessor, epochs: int = 400, batch_size: int = 32):
    data = prepare_dataset(processor)

    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_val   = data["X_val"]
    Y_val   = data["Y_val"]

    input_dim = X_train.shape[1]
    n_targets = Y_train.shape[1]

    model = build_mlp(input_dim, n_targets)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    metrics = evaluate_per_target(model, data)

    result = {
        "model": model,
        "history": history,
        "metrics": metrics,
        **data
    }
    return result


# ============================================================
# 9. MAIN
# ============================================================

if __name__ == "__main__":
    processor = MSTDBProcessor.from_csv("/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv")

    result = train_model(processor, epochs=400, batch_size=32)

    print("\n---------------- TRAINING COMPLETE ----------------")
    train_loss = result["history"].history["loss"]
    val_loss   = result["history"].history["val_loss"]
    print("Final train loss:", train_loss[-1])
    print("Final val   loss:", val_loss[-1])

    print("\nPer-target validation metrics (ALL targets, physical scale):")
    for name, vals in result["metrics"]["val"].items():
        mse = vals["MSE"]
        r2  = vals["R2"]
        print(f"{name:10s} | MSE: {mse:.4g} | R²: {r2:.4g}")

    # Example: properties at 1000 K using full predicted coefficients
    T_example = 1000.0
    props_list = predict_properties_at_T(processor, result, T_example)
    print(f"\nExample properties at T={T_example} K for first row:")
    print(props_list[0])
