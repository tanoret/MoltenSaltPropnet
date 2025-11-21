
# CatBoost / Ridge / GPR per target
# Parametric models + Temp-augmented k(T), cp(T) via Ridge
# Plotting & automatic diagnostics into visualisation/catboost
# should be worse than the metalayer catboost models

import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel

sns.set(style="whitegrid")


CSV_PATH = "/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv"


USE_TARGET_SCALING = False   # set True if you want to scale y per target
MAX_N_SPLITS_CV = 3          # maximum K for K-fold CV
MIN_SAMPLES = 8              # minimum rows required to train a target model

# Temperature switch for viscosity model (VFT below, Arrhenius above)
T_SWITCH = 850.0  # K

BASE_PLOT_DIR = "visualisation"
CATBOOST_PLOT_DIR = os.path.join(BASE_PLOT_DIR, "catboost")
os.makedirs(CATBOOST_PLOT_DIR, exist_ok=True)


def _savefig(filename):
    """Save plot into visualisation/catboost folder."""
    full_path = os.path.join(CATBOOST_PLOT_DIR, filename)
    plt.savefig(full_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot → {full_path}")


df = pd.read_csv(CSV_PATH)

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


df["comp"] = df.apply(extract_composition, axis=1)

elements = sorted({el for d in df["comp"] for el in d.keys()})

for el in elements:
    df[f"elem_{el}"] = df["comp"].apply(lambda d: d.get(el, 0.0))



drop_cols = ["System", "Mol Frac", "Composition", "comp"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Impute all feature NaNs to 0.0 (safe for element fractions, etc.)
X = X.fillna(0.0)

TARGETS = [
    "Melt(K)", "Boil(K)",
    "rho_a", "rho_b",
    "mu1_a", "mu1_b",          # Arrhenius params for viscosity
    "mu2_a", "mu2_b", "mu2_c", # VFT params for viscosity
    "k_a", "k_b",
    "cp_a", "cp_b",
]

# storage
models = {}
model_types = {}   # catboost / ridge / gpr per target
evals_results = {} # only for CatBoost
scalers = {}      



def choose_model_type(n_samples, target_name):
    """
    Decide which model to use based on sample count.
    - Large n: CatBoost
    - Medium n: Ridge
    - Tiny n: Gaussian Process
    You can tweak thresholds as you like.
    """
    if n_samples >= 200:
        return "catboost"
    elif n_samples >= 50:
        return "ridge"
    else:
        return "gpr"


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
print("Finished training all available target models (mixed types).")
print("============================================\n")


results = {}

print("\n=== MODEL PERFORMANCE ON ALL AVAILABLE DATA ===\n")

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
        "r2": r2,
        "mse": mse,
        "rel_mse_pct": relm,
        "y_true": y_true,
        "y_pred": y_pred,
    }

    print(f"{t}:")
    print(f"  model type   = {model_types[t]}")
    print(f"  R²           = {r2:.4f}")
    print(f"  MSE          = {mse:.6g}")
    print(f"  rel MSE (%)  = {relm:.3f}")
    print(f"  samples      = {len(y_true)}\n")


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


print("\n=== K-FOLD CROSS-VALIDATION (R²) ===\n")

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
        "mean_r2": mean_r2,
        "folds_r2": folds_r2,
    }

    print(f"{t}: model={model_type}, mean CV R² = {mean_r2:.4f}, folds = {[f'{v:.3f}' for v in folds_r2]}")


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
            "index": X_t.index[i],
            "target": t,
            "actual": true_val,
            "predicted": pred_val,
            "abs_error": abs(true_val - pred_val),
            "percent_error": 100 * abs(true_val - pred_val) / (abs(true_val) + 1e-12)
        })

pred_vs_actual = pd.DataFrame(rows)

print("\n=== PREDICTED vs ACTUAL (head) ===")
print(pred_vs_actual.head())

pred_vs_actual.to_csv("predicted_vs_actual.csv", index=False)
print("Saved predicted_vs_actual.csv")


def plot_actual_vs_predicted(df_pa, target, save=True):
    d = df_pa[df_pa["target"] == target]
    if len(d) == 0:
        print(f"No data for {target}")
        return

    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=d, x="actual", y="predicted", alpha=0.6)
    lo = min(d["actual"].min(), d["predicted"].min())
    hi = max(d["actual"].max(), d["predicted"].max())
    plt.plot([lo, hi], [lo, hi], "r--")

    plt.title(f"Actual vs Predicted – {target}")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()

    if save:
        _savefig(f"actual_vs_pred_{target}.png")

    plt.show()


def plot_residuals(df_pa, target, save=True):
    d = df_pa[df_pa["target"] == target].copy()
    if len(d) == 0:
        print(f"No data for {target}")
        return

    d["residual"] = d["predicted"] - d["actual"]

    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=d, x="actual", y="residual", alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")

    plt.title(f"Residuals – {target}")
    plt.xlabel("Actual")
    plt.ylabel("Residual")
    plt.tight_layout()

    if save:
        _savefig(f"residuals_{target}.png")

    plt.show()


def plot_error_distribution(df_pa, target, percent=False, save=True):
    d = df_pa[df_pa["target"] == target]
    if len(d) == 0:
        print(f"No data for {target}")
        return

    plt.figure(figsize=(6, 4))
    values = d["percent_error"] if percent else d["abs_error"]
    sns.histplot(values, bins=30, kde=True)

    plt.xlabel("Percent Error (%)" if percent else "Absolute Error")
    plt.title(f"Error Distribution – {target}")
    plt.tight_layout()

    if save:
        suffix = "pct" if percent else "abs"
        _savefig(f"error_dist_{target}_{suffix}.png")

    plt.show()


def plot_all_targets_grid(df_pa, targets, cols=4, save=True):
    n = len(targets)
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols * 4, rows * 3.2))

    for i, t in enumerate(targets):
        d = df_pa[df_pa["target"] == t]
        if len(d) == 0:
            continue

        plt.subplot(rows, cols, i + 1)
        sns.scatterplot(data=d, x="actual", y="predicted", alpha=0.6)
        lo = min(d["actual"].min(), d["predicted"].min())
        hi = max(d["actual"].max(), d["predicted"].max())
        plt.plot([lo, hi], [lo, hi], "r--")
        plt.title(t)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")

    plt.tight_layout()
    if save:
        _savefig("all_targets_grid.png")

    plt.show()


def plot_confusion_style_heatmap(df_pa, target, bins=40, save=True):
    d = df_pa[df_pa["target"] == target]
    if len(d) == 0:
        print(f"No data for {target}")
        return

    actual = d["actual"].values
    pred   = d["predicted"].values

    vmin = min(actual.min(), pred.min())
    vmax = max(actual.max(), pred.max())

    plt.figure(figsize=(6, 5))
    plt.hist2d(actual, pred, bins=bins,
               range=[[vmin, vmax], [vmin, vmax]],
               cmap="viridis")
    plt.plot([vmin, vmax], [vmin, vmax], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Confusion-style Heatmap – {target}")
    plt.colorbar(label="Count")
    plt.tight_layout()

    if save:
        _savefig(f"confusion_heatmap_{target}.png")

    plt.show()
# ============================================
# 15. FEATURE IMPORTANCE & SHAP PLOTS (CatBoost)
# ============================================

import shap

def plot_feature_importance_catboost(model, feature_names, target, save=True):
    """Plot CatBoost feature importance."""
    importances = model.feature_importances_

    plt.figure(figsize=(8, 6))
    idx = np.argsort(importances)[::-1]
    plt.barh(np.array(feature_names)[idx][::-1], importances[idx][::-1])
    plt.title(f"Feature Importance – {target}")
    plt.xlabel("Importance")
    plt.tight_layout()

    if save:
        _savefig(f"feature_importance_{target}.png")

    plt.show()

def plot_shap_summary_catboost(model, X_sample, target, save=True):
    """Plot SHAP summary (global importance + distribution)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
        plot_size=(8, 6),
        max_display=20
    )

    if save:
        _savefig(f"shap_summary_{target}.png")

    plt.show()

def plot_shap_dependence_catboost(model, X_sample, feature_name, target, save=True):
    """Optional: SHAP dependence for a single feature."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.dependence_plot(
        feature_name,
        shap_values,
        X_sample,
        show=False
    )

    if save:
        _savefig(f"shap_dependence_{target}_{feature_name}.png")

    plt.show()


def build_error_bin_matrix(df_pa, targets, bins=(0, 5, 10, 20, 50, 100)):
    labels = []
    for i in range(len(bins) - 1):
        labels.append(f"{bins[i]}–{bins[i+1]}%")
    labels.append(f">{bins[-1]}%")

    mat = pd.DataFrame(0, index=targets, columns=labels)

    for t in targets:
        d = df_pa[df_pa["target"] == t]
        if len(d) == 0:
            continue
        errs = d["percent_error"].values
        idxs = np.digitize(errs, bins)

        for idx in idxs:
            if idx < len(bins):
                col = labels[idx-1] if idx > 0 else labels[0]
            else:
                col = labels[-1]
            mat.loc[t, col] += 1

    return mat


def plot_error_bin_matrix(df_pa, targets, bins=(0, 5, 10, 20, 50, 100), save=True):
    mat = build_error_bin_matrix(df_pa, targets, bins=bins)

    plt.figure(figsize=(1.4 * len(mat.columns), 0.5 * len(mat.index) + 3))
    sns.heatmap(mat, annot=True, fmt="d", cmap="viridis", cbar_kws={"label": "Count"})
    plt.xlabel("Percent Error Bin")
    plt.ylabel("Target")
    plt.title("Error-bin Confusion Matrix")
    plt.tight_layout()

    if save:
        _savefig("error_bin_confusion_matrix.png")

    plt.show()


def plot_error_by_system(df_raw, df_pa, target, top_n_systems=10, percent=True, save=True):
    d = df_pa[df_pa["target"] == target]
    if len(d) == 0:
        print(f"No data for {target}")
        return

    merged = d.merge(df_raw[["System"]], left_on="index", right_index=True, how="left")

    err_col = "percent_error" if percent else "abs_error"
    ylabel  = "Percent Error (%)" if percent else "Absolute Error"

    top = merged["System"].value_counts().head(top_n_systems).index
    merged = merged[merged["System"].isin(top)]

    plt.figure(figsize=(max(8, 0.7 * len(top)), 4))
    sns.boxplot(data=merged, x="System", y=err_col)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Error by System – {target}")
    plt.ylabel(ylabel)
    plt.tight_layout()

    if save:
        _savefig(f"error_by_system_{target}.png")

    plt.show()


def plot_catboost_learning_curve(target, save=True):
    if target not in evals_results:
        print(f"No CatBoost evals found for {target}.")
        return

    ev = evals_results[target]
    train_loss = ev["learn"]["RMSE"]
    val_loss = ev["validation"]["RMSE"]

    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, label="Train RMSE")
    plt.plot(val_loss, label="Validation RMSE")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title(f"CatBoost Learning Curve – {target}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save:
        _savefig(f"learning_curve_{target}.png")

    plt.show()


def generate_all_plots(df_raw, pred_df, targets, top_n_systems=10):
    """
    Generate all diagnostic plots for all targets and save them
    into visualisation/catboost.
    """
    print("\n=== Generating per-target plots ===")
    for t in targets:
        d = pred_df[pred_df["target"] == t]
        if len(d) == 0:
            print(f"[{t}] – no data, skipping.")
            continue

        print(f"[{t}] – plotting...")

        plot_actual_vs_predicted(pred_df, t, save=True)
        plot_residuals(pred_df, t, save=True)
        plot_error_distribution(pred_df, t, percent=False, save=True)
        plot_error_distribution(pred_df, t, percent=True,  save=True)
        plot_confusion_style_heatmap(pred_df, t, bins=40, save=True)
        plot_error_by_system(df_raw, pred_df, t,
                             top_n_systems=top_n_systems,
                             percent=True,
                             save=True)

        if model_types.get(t) == "catboost":
            plot_catboost_learning_curve(t, save=True)

    print("\n=== Generating global plots (all targets) ===")
    plot_all_targets_grid(pred_df, targets, cols=4, save=True)
    plot_error_bin_matrix(pred_df, targets, bins=(0, 5, 10, 20, 50, 100), save=True)
    print("\nAll plots generated and saved in 'visualisation/catboost'.")


if __name__ == "__main__":
    idx = 0
    T_example = 1000.0

    print(f"\n=== Parameter + derived property prediction for row {idx} at T={T_example} K ===")
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
    print(f"Augmented Ridge k(T={T_example} K) for row {idx}: {k_pred}")

    print("\nFitting cp(T) Ridge model on temperature-augmented data...")
    ridge_cp, scaler_cp, feat_names_cp = build_cpT_model()
    cp_pred = predict_propT_for_row(ridge_cp, scaler_cp, feat_names_cp, X.iloc[[idx]], T_example)
    print(f"Augmented Ridge cp(T={T_example} K) for row {idx}: {cp_pred}")

    print("\nGenerating all diagnostic plots...")
    generate_all_plots(df, pred_vs_actual, TARGETS, top_n_systems=10)
