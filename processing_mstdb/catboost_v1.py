"""Different version of Catboost just with simpler logic, no meta-net, mixed model types. 
Really more like ML style I know that's not pure CatBoost but whatever."""


import os
import json
import re
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

CSV_PATH = "/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv"

USE_TARGET_SCALING = False   # set True if you want to scale y per target
MAX_N_SPLITS_CV = 3          # maximum K for K-fold CV
MIN_SAMPLES = 8              # minimum rows required to train a target model

# Temperature switch for viscosity model (VFT below, Arrhenius above)
T_SWITCH = 850.0  # K

TARGETS = [
    "Melt(K)", "Boil(K)",
    "rho_a", "rho_b",
    "mu1_a", "mu1_b",          # Arrhenius params for viscosity
    "mu2_a", "mu2_b", "mu2_c", # VFT params for viscosity
    "k_a", "k_b",
    "cp_a", "cp_b",
]

EVAL_ROOT = "evaluate_modelperformance"
EVAL_CATBOOST_V1_DIR = os.path.join(EVAL_ROOT, "catboost_v1")

BEST_ROOT = "best_models"
BEST_CATBOOST_V1_DIR = os.path.join(BEST_ROOT, "catboost_v1")

os.makedirs(EVAL_CATBOOST_V1_DIR, exist_ok=True)
os.makedirs(BEST_CATBOOST_V1_DIR, exist_ok=True)

# storage
models = {}
model_types = {}   # catboost / ridge / gpr per target
evals_results = {} # only for CatBoost
scalers = {}

# -------------------------------------------------
# HELPERS
# -------------------------------------------------

def parse_composition_str(x):
    """Convert the composition JSON-like string to a Python dict, if possible."""
    if not isinstance(x, str) or x.strip() == "":
        return None
    try:
        return json.loads(x.replace("'", '"'))
    except Exception:
        return None


def parse_formula(compound: str):
    """
    Parse simple inorganic formulas like NaCl, CaF2, Al2O3.
    Parentheses are NOT supported; this is fine for simple salts.
    """
    if not isinstance(compound, str):
        return {}
    out = {}
    matches = re.findall(r"([A-Z][a-z]*)(\d*)", compound)
    for el, num in matches:
        n = int(num) if num else 1
        out[el] = out.get(el, 0) + n
    return out


def normalize_counts_to_fractions(d: dict):
    """Convert element counts to fractions that sum to 1."""
    if not d:
        return {}
    total = float(sum(d.values()))
    if total <= 0:
        return d
    return {k: v / total for k, v in d.items()}


def extract_composition(row):
    """
    Prefer the Composition column (if parseable),
    otherwise fall back to parsing the System formula.
    """
    comp = parse_composition_str(row.get("Composition", None))
    if comp is None:
        comp = parse_formula(row.get("System", ""))
    return normalize_counts_to_fractions(comp)


def rel_mse_pct(y_true, y_pred):
    """Relative MSE in %, normalized by mean(y_true^2)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    denom = np.mean(y_true ** 2)
    if denom < 1e-12:
        denom = 1e-12
    return 100.0 * mse / denom


def choose_model_type(n_samples, target_name):
    """
    Decide which model to use based on sample count.
    - Large n: CatBoost
    - Medium n: Ridge
    - Tiny n: Gaussian Process
    """
    if n_samples >= 200:
        return "catboost"
    elif n_samples >= 50:
        return "ridge"
    else:
        return "gpr"


# -------------------------------------------------
# DATA & FEATURES
# -------------------------------------------------

df = pd.read_csv(CSV_PATH)

df["comp"] = df.apply(extract_composition, axis=1)
elements = sorted({el for d in df["comp"] for el in d.keys()})

for el in elements:
    df[f"elem_{el}"] = df["comp"].apply(lambda d: d.get(el, 0.0))

drop_cols = ["System", "Mol Frac", "Composition", "comp"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Impute all feature NaNs to 0.0 (safe for element fractions, etc.)
X = X.fillna(0.0)

# -------------------------------------------------
# TRAIN PER-TARGET MODELS
# -------------------------------------------------

for t in TARGETS:
    if t not in df.columns:
        print(f"Skipping {t}: column not found.")
        continue

    mask = df[t].notna()
    n = int(mask.sum())
    if n < MIN_SAMPLES:
        print(f"Skipping {t}: not enough data ({n} rows).")
        continue

    model_type = choose_model_type(n, t)
    model_types[t] = model_type
    print(f"Training model for: {t} (n={n}, model={model_type})")

    X_t = X[mask]
    y_t = df.loc[mask, t]

    if USE_TARGET_SCALING:
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(y_t.values.reshape(-1, 1)).ravel()
        scalers[t] = scaler
    else:
        y_scaled = y_t.values

    if model_type == "catboost":
        X_train, X_val, y_train, y_val = train_test_split(
            X_t, y_scaled, test_size=0.2, random_state=42
        )

        model = CatBoostRegressor(
            loss_function="RMSE",
            depth=8,
            learning_rate=0.03,
            iterations=1200,
            eval_metric="RMSE",
            random_seed=42,
            verbose=False
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        evals_results[t] = model.get_evals_result()

    elif model_type == "ridge":
        model = Ridge(alpha=1.0)
        model.fit(X_t, y_scaled)

    elif model_type == "gpr":
        kernel = C(1.0, (1e-2, 1e3)) * Matern(length_scale=1.0, nu=1.5) \
                 + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=42
        )
        model.fit(X_t, y_scaled)

    models[t] = model

print("\n============================================")
print("Finished training all available target models (catboostv2, mixed types).")
print("============================================\n")

# -------------------------------------------------
# EVALUATION ON ALL DATA
# -------------------------------------------------

results = {}

print("\n=== catboostv2 MODEL PERFORMANCE ON ALL AVAILABLE DATA ===\n")

for t in TARGETS:
    model = models.get(t, None)
    if model is None:
        print(f"Skipping {t}: model not trained.")
        continue

    mask = df[t].notna()
    X_t = X[mask]
    y_true = df.loc[mask, t].values

    y_pred_scaled = model.predict(X_t)

    if USE_TARGET_SCALING and t in scalers:
        scaler = scalers[t]
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    else:
        y_pred = y_pred_scaled

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    relm = rel_mse_pct(y_true, y_pred)

    results[t] = {
        "model_type": model_types[t],
        "r2": float(r2),
        "mse": float(mse),
        "rel_mse_pct": float(relm),
        "n_samples": int(len(y_true)),
    }

    print(f"{t}:")
    print(f"  model type   = {model_types[t]}")
    print(f"  R²           = {r2:.4f}")
    print(f"  MSE          = {mse:.6g}")
    print(f"  rel MSE (%)  = {relm:.3f}")
    print(f"  samples      = {len(y_true)}\n")


# -------------------------------------------------
# K-FOLD CV
# -------------------------------------------------

def crossval_for_target(X_t, y_t, model_type, use_scaling=False, max_splits=MAX_N_SPLITS_CV):
    """Re-train the same model type in K-fold CV for a given target."""
    n_samples = len(X_t)
    n_splits = min(max_splits, n_samples)
    if n_splits < 2:
        return None, []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X_t):
        X_tr, X_va = X_t.iloc[train_idx], X_t.iloc[val_idx]
        y_tr, y_va = y_t.iloc[train_idx], y_t.iloc[val_idx]

        if use_scaling:
            scaler = StandardScaler()
            y_tr_s = scaler.fit_transform(y_tr.values.reshape(-1, 1)).ravel()
            y_va_s = scaler.transform(y_va.values.reshape(-1, 1)).ravel()
        else:
            y_tr_s = y_tr.values
            y_va_s = y_va.values

        if model_type == "catboost":
            m = CatBoostRegressor(
                loss_function="RMSE",
                depth=8,
                learning_rate=0.03,
                iterations=800,
                eval_metric="RMSE",
                random_seed=42,
                verbose=False
            )
            m.fit(X_tr, y_tr_s, eval_set=(X_va, y_va_s), verbose=False)
        elif model_type == "ridge":
            m = Ridge(alpha=1.0)
            m.fit(X_tr, y_tr_s)
        elif model_type == "gpr":
            kernel = C(1.0, (1e-2, 1e3)) * Matern(length_scale=1.0, nu=1.5) \
                     + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
            m = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=1,
                random_state=42
            )
            m.fit(X_tr, y_tr_s)
        else:
            continue

        y_pred_s = m.predict(X_va)
        if use_scaling:
            y_pred = scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
        else:
            y_pred = y_pred_s

        scores.append(r2_score(y_va, y_pred))

    return float(np.mean(scores)), scores


print("\n=== catboostv2 K-FOLD CROSS-VALIDATION (R²) ===\n")

cv_results = {}

for t in TARGETS:
    if t not in df.columns or t not in models:
        continue
    model_type = model_types.get(t, None)
    if model_type is None:
        continue

    mask = df[t].notna()
    n = int(mask.sum())
    if n < MIN_SAMPLES:
        continue

    X_t = X[mask]
    y_t = df.loc[mask, t]

    mean_r2, folds_r2 = crossval_for_target(
        X_t, y_t, model_type=model_type, use_scaling=USE_TARGET_SCALING
    )
    if mean_r2 is None:
        continue

    cv_results[t] = {
        "model_type": model_type,
        "mean_r2": float(mean_r2),
        "folds_r2": [float(v) for v in folds_r2],
    }

    print(f"{t}: model={model_type}, mean CV R² = {mean_r2:.4f}, folds = {[f'{v:.3f}' for v in folds_r2]}")


# -------------------------------------------------
# PREDICTED vs ACTUAL TABLE
# -------------------------------------------------

rows = []

for t in TARGETS:
    model = models.get(t)
    if model is None:
        continue

    mask = df[t].notna()
    X_t = X[mask]
    y_true = df.loc[mask, t].values

    y_pred_scaled = model.predict(X_t)

    # if target scaling was used, invert it
    if USE_TARGET_SCALING and t in scalers:
        scaler = scalers[t]
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    else:
        y_pred = y_pred_scaled

    for i, (true_val, pred_val) in enumerate(zip(y_true, y_pred)):
        rows.append({
            "index": int(X_t.index[i]),
            "target": t,
            "actual": float(true_val),
            "predicted": float(pred_val),
            "abs_error": float(abs(true_val - pred_val)),
            "percent_error": float(100 * abs(true_val - pred_val) / (abs(true_val) + 1e-12))
        })

pred_vs_actual = pd.DataFrame(rows)

print("\n=== catboostv2 PREDICTED vs ACTUAL (head) ===")
print(pred_vs_actual.head())

PVA_PATH = os.path.join(EVAL_CATBOOST_V1_DIR, "predicted_vs_actual_catboostv2.csv")
pred_vs_actual.to_csv(PVA_PATH, index=False)
print(f"Saved {PVA_PATH}")

# -------------------------------------------------
# PHYSICS / PARAMETRIC HELPERS
# -------------------------------------------------

def predict_parameters(row_df):
    """
    Predict raw target parameters for a 1-row DataFrame of features.
    Returns a dict of parameter_name -> value (or None if no model).
    """
    out = {}
    for t in TARGETS:
        model = models.get(t, None)
        if model is None:
            out[t] = None
            continue

        y_pred_s = model.predict(row_df)
        if USE_TARGET_SCALING and t in scalers:
            scaler = scalers[t]
            y_pred = scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()[0]
        else:
            y_pred = y_pred_s[0]
        out[t] = float(y_pred)
    return out


def derive_physical_properties(params, T):
    """
    Use standard fit forms to derive physical properties at temperature T (K).

    - Density:   rho(T)  = rho_a + rho_b * T
    - Viscosity:
        Arrhenius: ln(mu)    = mu1_a + mu1_b / T
        VFT:       log10(mu) = mu2_a + mu2_b / (T - mu2_c)
        Combined mu(T): VFT below T_SWITCH, Arrhenius at/above T_SWITCH
    - k(T)  = k_a + k_b * T
    - cp(T) = cp_a + cp_b * T
    Returns dict of {name: value or None}
    """

    out = {}

    # ----- Density -----
    if params.get("rho_a") is not None and params.get("rho_b") is not None:
        out["rho(T)"] = params["rho_a"] + params["rho_b"] * T
    else:
        out["rho(T)"] = None

    # ----- Viscosity: Arrhenius -----
    mu_arr = None
    if params.get("mu1_a") is not None and params.get("mu1_b") is not None:
        A = params["mu1_a"]
        B = params["mu1_b"]
        exponent = A + B / T
        exponent = np.clip(exponent, -50, 50)
        mu_arr = float(np.exp(exponent))
    out["mu_arr(T)"] = mu_arr

    # ----- Viscosity: VFT -----
    mu_vft = None
    if (
        params.get("mu2_a") is not None
        and params.get("mu2_b") is not None
        and params.get("mu2_c") is not None
    ):
        A_vft = params["mu2_a"]
        B_vft = params["mu2_b"]
        C_vft = params["mu2_c"]

        denom = T - C_vft
        if abs(denom) < 1e-6:
            denom = np.sign(denom) * 1e-6 if denom != 0 else 1e-6

        log10_mu = A_vft + B_vft / denom
        exponent_vft = np.log(10.0) * log10_mu
        exponent_vft = np.clip(exponent_vft, -50, 50)

        mu_vft = float(np.exp(exponent_vft))
    out["mu_vft(T)"] = mu_vft

    # ----- Combined viscosity with T-switch -----
    if T < T_SWITCH and mu_vft is not None:
        mu_combined = mu_vft
    elif T >= T_SWITCH and mu_arr is not None:
        mu_combined = mu_arr
    else:
        mu_combined = mu_arr if mu_arr is not None else mu_vft
    out["mu(T)"] = mu_combined

    # ----- Thermal conductivity -----
    if params.get("k_a") is not None and params.get("k_b") is not None:
        out["k(T)"] = params["k_a"] + params["k_b"] * T
    else:
        out["k(T)"] = None

    # ----- Heat capacity -----
    if params.get("cp_a") is not None and params.get("cp_b") is not None:
        out["cp(T)"] = params["cp_a"] + params["cp_b"] * T
    else:
        out["cp(T)"] = None

    return out


def predict_all_for_index(idx, T=1000.0):
    """
    Convenience function:
    - takes a row index from df
    - uses its feature row X[idx]
    - predicts parameters
    - derives physical properties at temperature T
    """
    row_df = X.iloc[[idx]]
    params = predict_parameters(row_df)
    phys = derive_physical_properties(params, T)
    return params, phys


def make_temp_augmented_dataset_for_property(df_in, X_in, param_cols, temp_grid, prop_name, formula_fn):
    """
    Build an augmented dataset for a temperature-dependent property.

    df_in      : original DataFrame
    X_in       : feature DataFrame (same index as df)
    param_cols : list of columns needed (e.g. ["k_a", "k_b"])
    temp_grid  : list/array of temperatures (K)
    prop_name  : name of the property, e.g. "k(T)"
    formula_fn : function(params_dict, T) -> property value

    Returns: X_aug, y_aug where
        X_aug has columns of X plus a new column 'T'
        y_aug is the property at that T
    """
    rows_X = []
    rows_y = []

    mask = df_in[param_cols].notna().all(axis=1)
    df_sub = df_in[mask]
    X_sub = X_in[mask]

    for idx, row in df_sub.iterrows():
        params = {c: row[c] for c in param_cols}
        x_base = X_sub.loc[idx].values

        for T in temp_grid:
            y_val = formula_fn(params, T)
            if y_val is None:
                continue

            feat_with_T = np.concatenate([x_base, [T]])
            rows_X.append(feat_with_T)
            rows_y.append(y_val)

    feature_names = list(X_in.columns) + ["T"]
    X_aug = pd.DataFrame(rows_X, columns=feature_names)
    y_aug = np.array(rows_y)

    print(f"Augmented dataset for {prop_name}: {len(y_aug)} samples from {df_sub.shape[0]} original rows.")
    return X_aug, y_aug


def k_formula(params, T):
    """k(T) = k_a + k_b * T"""
    if params.get("k_a") is None or params.get("k_b") is None:
        return None
    return params["k_a"] + params["k_b"] * T


def cp_formula(params, T):
    """cp(T) = cp_a + cp_b * T"""
    if params.get("cp_a") is None or params.get("cp_b") is None:
        return None
    return params["cp_a"] + params["cp_b"] * T


def build_kT_model(temp_grid=None):
    """
    Train a Ridge Regression model to predict k(T) directly from augmented (X + T).
    """
    if temp_grid is None:
        temp_grid = np.linspace(700, 1300, 7)

    X_k_aug, y_k_aug = make_temp_augmented_dataset_for_property(
        df_in=df,
        X_in=X,
        param_cols=["k_a", "k_b"],
        temp_grid=temp_grid,
        prop_name="k(T)",
        formula_fn=k_formula
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_k_aug)

    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y_k_aug)

    return model, scaler, X_k_aug.columns


def build_cpT_model(temp_grid=None):
    """
    Train a Ridge Regression model to predict cp(T) directly from augmented (X + T).
    """
    if temp_grid is None:
        temp_grid = np.linspace(700, 1300, 7)

    X_cp_aug, y_cp_aug = make_temp_augmented_dataset_for_property(
        df_in=df,
        X_in=X,
        param_cols=["cp_a", "cp_b"],
        temp_grid=temp_grid,
        prop_name="cp(T)",
        formula_fn=cp_formula
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cp_aug)

    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y_cp_aug)

    return model, scaler, X_cp_aug.columns


def predict_propT_for_row(model, scaler, feature_names, X_row, T):
    """
    Predict a T-dependent property using a Ridge model trained on augmented X+T.
    """
    x_base = X_row.values[0]
    feat_with_T = np.concatenate([x_base, [T]])

    df_feat = pd.DataFrame([feat_with_T], columns=feature_names)

    x_scaled = scaler.transform(df_feat)
    y_pred = float(model.predict(x_scaled)[0])

    # Optional physical constraint: enforce non-negative values
    if y_pred < 0:
        y_pred = 0.0

    return y_pred


# -------------------------------------------------
# SAVE JSON ARTIFACTS + DEMO
# -------------------------------------------------

evals_path = os.path.join(EVAL_CATBOOST_V1_DIR, "catboostv2_evals_results.json")
cv_path = os.path.join(EVAL_CATBOOST_V1_DIR, "catboostv2_cv_results.json")
perf_path = os.path.join(EVAL_CATBOOST_V1_DIR, "catboostv2_performance_summary.json")

with open(evals_path, "w") as f:
    json.dump(evals_results, f)

with open(cv_path, "w") as f:
    json.dump(cv_results, f, indent=2)

with open(perf_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved {evals_path}")
print(f"Saved {cv_path}")
print(f"Saved {perf_path}")


if __name__ == "__main__":
    idx = 0
    T_example = 1000.0

    print(f"\n=== catboostv2: Parameter + derived property prediction for row {idx} at T={T_example} K ===")
    params_pred, phys_pred = predict_all_for_index(idx, T=T_example)

    print("\nPredicted parameters:")
    for k, v in params_pred.items():
        print(f"  {k}: {v}")

    print(f"\nDerived physical properties at T = {T_example} K:")
    for k, v in phys_pred.items():
        print(f"  {k}: {v}")

    print("\nFitting k(T) Ridge model on temperature-augmented data...")
    ridge_k, scaler_k, feat_names_k = build_kT_model()
    k_pred = predict_propT_for_row(ridge_k, scaler_k, feat_names_k, X.iloc[[idx]], T_example)
    print(f"catboostv2 augmented Ridge k(T={T_example} K) for row {idx}: {k_pred}")

    print("\nFitting cp(T) Ridge model on temperature-augmented data...")
    ridge_cp, scaler_cp, feat_names_cp = build_cpT_model()
    cp_pred = predict_propT_for_row(ridge_cp, scaler_cp, feat_names_cp, X.iloc[[idx]], T_example)
    print(f"catboostv2 augmented Ridge cp(T={T_example} K) for row {idx}: {cp_pred}")

    print("\nDone training catboostv2 and exporting artifacts.")
