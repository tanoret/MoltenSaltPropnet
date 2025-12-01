import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import clone

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)

# HistGradientBoostingRegressor (modern GB, fast, supports missing values in X)
from sklearn.ensemble import HistGradientBoostingRegressor

# -------------------------------------------------------------------
# 0. IMPORT YOUR MSTDBProcessor
# -------------------------------------------------------------------

# Adjust if your repo structure is different
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from processing_mstdb.processor import MSTDBProcessor


# -------------------------------------------------------------------
# 1. ROBUST compute_composition (handles numeric Mol Frac, Pure Salt, etc.)
#    (same logic style as in your DL scripts)
# -------------------------------------------------------------------

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


# -------------------------------------------------------------------
# 2. DATA PREPARATION
# -------------------------------------------------------------------

ALL_TARGET_COLS = [
    "Melt(K)", "Boil(K)",
    "rho_a", "rho_b",
    "mu1_a", "mu1_b",
    "mu2_a", "mu2_b", "mu2_c",
    "k_a", "k_b",
    "cp_a", "cp_b",
]


def prepare_dataset(processor: MSTDBProcessor):
    df = processor.df.copy()

    # Elemental composition from System + Mol Frac
    df["Composition"] = df.apply(
        lambda row: processor.compute_composition(row, composition_type="elements"),
        axis=1
    )

    # Collect all elements from processor
    elements = sorted(processor.predefined_elements)

    # Element fraction features
    for el in elements:
        df[f"elem_{el}"] = df["Composition"].apply(lambda d: d.get(el, 0.0))

    # Simple extra features
    df["is_mixture"] = df["System"].apply(lambda s: 1.0 if "-" in str(s) else 0.0)
    df["n_elements"] = df["Composition"].apply(lambda d: float(len(d)))

    # Ensure all target columns exist
    for col in ALL_TARGET_COLS:
        if col not in df.columns:
            df[col] = np.nan

    target_cols = list(ALL_TARGET_COLS)

    # Feature matrix (no scaling needed for trees/boosting)
    feat_cols = ["Mol Mass"] + [f"elem_{el}" for el in elements] + ["is_mixture", "n_elements"]
    X = df[feat_cols].astype("float32").fillna(0.0).values

    # Targets (physical values, with NaNs)
    Y = df[target_cols].astype("float32").values

    # Train/val split (same split for all targets)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    return {
        "df": df,
        "feat_cols": feat_cols,
        "target_cols": target_cols,
        "elements": elements,
        "X_train": X_train,
        "X_val": X_val,
        "Y_train": Y_train,
        "Y_val": Y_val,
    }


# -------------------------------------------------------------------
# 3. MODEL FACTORY
# -------------------------------------------------------------------

def get_base_model(model_type: str):
    """
    model_type ∈ {"random_forest", "extra_trees", "grad_boost", "hist_grad_boost"}
    """
    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        )
    elif model_type == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=400,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        )
    elif model_type == "grad_boost":
        return GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
    elif model_type == "hist_grad_boost":
        return HistGradientBoostingRegressor(
            max_depth=None,
            learning_rate=0.1,
            max_iter=500,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# -------------------------------------------------------------------
# 4. TRAIN ONE MODEL PER TARGET
# -------------------------------------------------------------------

def train_per_target_models(data, model_type="random_forest"):
    """
    data: output of prepare_dataset
    model_type: one of "random_forest", "extra_trees", "grad_boost", "hist_grad_boost"
    Returns:
      models: dict target_name -> fitted estimator or None (if not enough data)
      metrics: dict target_name -> {"train": {...}, "val": {...}}
    """
    base_model = get_base_model(model_type)

    X_train = data["X_train"]
    X_val   = data["X_val"]
    Y_train = data["Y_train"]
    Y_val   = data["Y_val"]
    target_cols = data["target_cols"]

    models = {}
    metrics = {}

    print(f"\n======== Training {model_type} per target ========")

    for j, name in enumerate(target_cols):
        y_tr = Y_train[:, j]
        y_val = Y_val[:, j]

        # Use only samples where this target is not NaN
        mask_tr = ~np.isnan(y_tr)
        mask_val = ~np.isnan(y_val)

        n_tr = mask_tr.sum()
        n_val = mask_val.sum()

        if n_tr < 8 or n_val < 2:
            print(f"Skipping {name:10s}: not enough labeled data (train={n_tr}, val={n_val})")
            models[name] = None
            metrics[name] = {
                "train": {"MSE": np.nan, "R2": np.nan},
                "val":   {"MSE": np.nan, "R2": np.nan},
            }
            continue

        model = clone(base_model)
        model.fit(X_train[mask_tr], y_tr[mask_tr])

        # Predictions
        y_tr_pred = model.predict(X_train[mask_tr])
        y_val_pred = model.predict(X_val[mask_val])

        mse_tr = mean_squared_error(y_tr[mask_tr], y_tr_pred)
        r2_tr  = r2_score(y_tr[mask_tr], y_tr_pred)

        mse_val = mean_squared_error(y_val[mask_val], y_val_pred)
        r2_val  = r2_score(y_val[mask_val], y_val_pred)

        models[name] = model
        metrics[name] = {
            "train": {"MSE": mse_tr, "R2": r2_tr},
            "val":   {"MSE": mse_val, "R2": r2_val},
        }

        print(f"{name:10s} | train R²={r2_tr: .3f}, val R²={r2_val: .3f}, val MSE={mse_val: .4g}")

    return models, metrics


# -------------------------------------------------------------------
# 5. MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":

    MODEL_TYPES = [
        "random_forest",
        "extra_trees",
        "grad_boost",
        "hist_grad_boost",
    ]

    processor = MSTDBProcessor.from_csv("/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv")
    data = prepare_dataset(processor)

    results = {}   # model_type → metrics

    print("\n==============================================")
    print("     Evaluating ALL classical models")
    print("==============================================")

    for mtype in MODEL_TYPES:
        print(f"\n\n>>>>>>>>>> Training model: {mtype} <<<<<<<<<<")

        models, metrics = train_per_target_models(data, model_type=mtype)
        results[mtype] = metrics

    # -------------------------------------------------------------
    # SUMMARY TABLE
    # -------------------------------------------------------------

    target_cols = data["target_cols"]

    print("\n\n================== FINAL SUMMARY ==================\n")

    # Header
    header = "Model".ljust(15) + "Mean R²   "
    for t in target_cols:
        header += t.ljust(12)
    print(header)
    print("-" * len(header))

    # Rows per model
    for mtype in MODEL_TYPES:
        metrics = results[mtype]

        r2_vals = []
        row = mtype.ljust(15)

        # Compute mean R2 across available targets
        for t in target_cols:
            r2 = metrics[t]["val"]["R2"]
            r2_vals.append(r2)
        mean_r2 = np.nanmean(r2_vals)

        row += f"{mean_r2:8.3f}   "

        for r2 in r2_vals:
            row += f"{r2:8.3f}   "

        print(row)

    print("\n====================================================\n")
