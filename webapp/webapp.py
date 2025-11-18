import os
import sys
from pathlib import Path
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Add project to path
# PROJECT_PATH = os.path.abspath(os.path.join(os.getcwd(), '..'))
# PROJECT_PATH = pathlib.Path(__file__).resolve().parent.parent
# if PROJECT_PATH not in sys.path:
    # sys.path.insert(0, PROJECT_PATH)

# Import processing and trainer modules
# from processing_saltdblean.processor import SALTDBLEANProcessor
# from processing_saltdblean.trainer import AIModelTrainer
# from processing_saltdblean.resnet_trainer import ResNetMetaTrainer, TARGETS as RESNET_TARGETS, DERIVED_PROPS as RESNET_DERIVED_PROPS
# from processing_saltdblean.kan_trainer import KANMetaTrainer, TARGETS as KAN_TARGETS, DERIVED_PROPS as KAN_DERIVED_PROPS
# from processing_saltdblean.snn_trainer import SNNMetaTrainer, TARGETS as SNN_TARGETS, DERIVED_PROPS as SNN_DERIVED_PROPS

from processing_saltdblean.processor import SALTDBLEANProcessor
from processing_saltdblean.trainer   import AIModelTrainer
from processing_saltdblean.resnet_trainer import (
    ResNetMetaTrainer, TARGETS as RESNET_TARGETS, DERIVED_PROPS as RESNET_DERIVED_PROPS
)
from processing_saltdblean.kan_trainer import (
    KANMetaTrainer, TARGETS as KAN_TARGETS, DERIVED_PROPS as KAN_DERIVED_PROPS
)
from processing_saltdblean.snn_trainer import (
    SNNMetaTrainer, TARGETS as SNN_TARGETS, DERIVED_PROPS as SNN_DERIVED_PROPS
)

# Universal gas constant
R = 8.314

def derive_properties(coeffs, T):
    """
    Compute thermophysical properties from fit coefficients at temperature T.
    Missing coefficients default to zero.
    """
    out = {}
    # Density: rho = a - b * T
    a_rho = coeffs.get("rho_a", 0.0)
    b_rho = coeffs.get("rho_b", 0.0)
    out["rho"] = a_rho - b_rho * T

    # Viscosity (Arrhenius): muA = a * exp(b / (R*T))
    a1 = coeffs.get("mu1_a", 0.0)
    b1 = coeffs.get("mu1_b", 0.0)
    out["muA"] = a1 * math.exp(b1 / (R * T))

    # Thermal conductivity: k = a + b * T
    a_k = coeffs.get("k_a", 0.0)
    b_k = coeffs.get("k_b", 0.0)
    out["k"] = a_k + b_k * T

    # Heat capacity: cp = a + b * T + c / T^2
    a_cp = coeffs.get("cp_a", 0.0)
    b_cp = coeffs.get("cp_b", 0.0)
    c_cp = coeffs.get("cp_c", 0.0)
    out["cp"] = a_cp + b_cp * T + c_cp / (T**2)

    return out

# Discover available CSV files
PROJECT_PATH = Path(__file__).resolve().parents[1]   # …/MoltenSaltPropnet
DATA_DIR     = PROJECT_PATH / "data"
csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])

# Streamlit sidebar: dataset selection
st.sidebar.title("Data & Composition Options")
dataset = st.sidebar.selectbox("Select dataset:", csv_files)
composition_type = st.sidebar.radio(
    "Composition type:",
    options=["elements", "compounds", "both"]
)

# Load processor into session state
if 'processor' not in st.session_state or st.session_state.get('dataset') != dataset:
    df_path = os.path.join(DATA_DIR, dataset)
    processor = SALTDBLEANProcessor.from_csv(df_path)
    processor.df["Composition"] = processor.df.apply(
        lambda row: processor.compute_composition(
            row, composition_type=composition_type
        ),
        axis=1
    )
    st.session_state.processor = processor
    st.session_state.dataset = dataset

st.write(f"**Loaded**: {dataset} with {len(st.session_state.processor.df)} rows")

import streamlit as st
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Part 2: Data Filtering & Export UI
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.subheader("Filter Dataset")

processor = st.session_state.processor

# Ensure composition column is up-to-date
processor.df["Composition"] = processor.df.apply(
    lambda row: processor.compute_composition(
        row, composition_type=composition_type
    ), axis=1
)

# Build options based on composition type
if composition_type == "elements":
    opts = sorted(processor.predefined_elements)
elif composition_type == "compounds":
    opts = sorted(processor.predefined_compounds)
else:
    opts = sorted(processor.predefined_elements | processor.predefined_compounds)

include = st.sidebar.multiselect("Include components:", opts)

if st.sidebar.button("Apply Filter"):
    if not include:
        st.sidebar.warning("Select at least one component to filter.")
    else:
        if composition_type == "both":
            elems = [x for x in include if x in processor.predefined_elements]
            comps = [x for x in include if x in processor.predefined_compounds]
            inc_dict = {"elements": elems, "compounds": comps}
        else:
            inc_dict = {composition_type: include}
        flt = {"include": inc_dict}

        filtered_proc = processor.filter_by_components(flt)
        # drop duplicate columns
        filtered_proc.df = filtered_proc.df.loc[:, ~filtered_proc.df.columns.duplicated()]
        st.session_state.filtered_proc = filtered_proc
        st.session_state.include = include
        st.sidebar.success(f"Filtered to {len(filtered_proc.df)} rows.")

# Export filtered data
if "filtered_proc" in st.session_state:
    df_filtered = st.session_state.filtered_proc.df
    csv = df_filtered.to_csv(index=False).encode()
    st.sidebar.download_button(
        label="Export Filtered Data",
        data=csv,
        file_name="filtered_data.csv",
        mime="text/csv"
    )

# Main page preview
if "filtered_proc" in st.session_state:
    st.subheader("Filtered Dataset Preview")
    st.dataframe(st.session_state.filtered_proc.df.head(10))
else:
    st.info("No filter applied yet.")



# ─────────────────────────────────────────────────────────────────────────────
# Part 3: Model Training Section
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.subheader("Model Training")

# Ensure data is filtered before training
if "filtered_proc" not in st.session_state:
    st.sidebar.info("Filter dataset before training.")
else:

    # Embedding options
    emethod = st.sidebar.selectbox(
        "Embedding method:",
        options=[
            "none",
            "pca",
            "feature_hashing",
            "tsne",
            "low_variance",
            "nmf",
            "svd",
        ],
        format_func=lambda m: {
            "none": "None",
            "pca": "PCA",
            "feature_hashing": "Feature Hashing",
            "tsne": "t-SNE",
            "low_variance": "Low-Variance Filter",
            "nmf": "NMF",
            "svd": "SVD",
        }[m],
    )
    ncomp = st.sidebar.slider(
        "Embedding dimensions:",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
    )

    # Map for friendly model names
    model_map = {
        "sklearn": "Scikit‑Learn Polynomial Regressor",
        "resnet":  "ResNet + Meta‑Learning + Physics",
        "kan":     "Kernel‑Approximation Network (KAN) + Meta",
        "snn":     "Spiking Neural Network (SNN) + Meta"
    }
    model_key = st.sidebar.selectbox(
        "Select Model:",
        options=list(model_map.keys()),
        format_func=lambda x: model_map[x]
    )

    # Train button with progress spinner
    if st.sidebar.button("Train Model"):
        df_train = st.session_state.filtered_proc.df
        if df_train.empty:
            st.sidebar.error("Filtered dataset is empty.")
        else:
            with st.spinner("Training model, this may take several minutes..."):
                if model_key == "sklearn":
                    trainer = AIModelTrainer(
                        df_train,
                        embedding_method=emethod,
                        embedding_params={"n_components": ncomp})
                    trainer.train_all()
                elif model_key == "resnet":
                    trainer = ResNetMetaTrainer(
                            df_train,
                            RESNET_TARGETS,
                            RESNET_DERIVED_PROPS,
                            embedding_method=emethod,
                            n_components=ncomp)
                    trainer.train_base(); trainer.train_meta(); #trainer.train_joint()
                elif model_key == "kan":
                    trainer = KANMetaTrainer(
                            df_train,
                            RESNET_TARGETS,
                            RESNET_DERIVED_PROPS,
                            embedding_method=emethod,
                            n_components=ncomp)
                    trainer.train_base(); trainer.train_meta(); #trainer.train_joint()
                else:
                    trainer = SNNMetaTrainer(
                            df_train,
                            RESNET_TARGETS,
                            RESNET_DERIVED_PROPS,
                            embedding_method=emethod,
                            n_components=ncomp)
                    trainer.train_base(); trainer.train_meta(); #trainer.train_joint()
            # Persist trainer and model_key
            st.session_state.trainer = trainer
            st.session_state.model_key = model_key
            if hasattr(trainer, "evaluate"):                 # ResNet / KAN / SNN
                st.session_state.metrics = trainer.evaluate(return_dict=True)
            elif hasattr(trainer, "summary_metrics"):        # Scikit-Learn trainer
                st.session_state.metrics = trainer.summary_metrics()
            else:                                            # Fallback – nothing to show
                st.session_state.metrics = {}
            st.sidebar.success(f"Training complete: {model_map[model_key]}")

# On main page, show status
if 'trainer' in st.session_state:
    st.subheader("Model Status")
    st.write(f"Trained model: **{model_map[st.session_state.model_key]}** on {len(st.session_state.filtered_proc.df)} samples.")
else:
    st.info("No trained model in session.")

if 'metrics' in st.session_state and st.session_state.metrics:
    st.subheader("Training / Validation Metrics")
    m = st.session_state.metrics

    dfm = (pd.DataFrame(m["per_target"])
             .T.rename(columns={"MSE_pct": "MSE (%)"})
             .sort_values("MSE (%)"))
    st.dataframe(dfm.style.format({"MSE (%)":"{:.2f}", "R2":"{:+.3f}"}))

import streamlit as st
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Part 4: Prediction Section
# ─────────────────────────────────────────────────────────────────────────────
if 'trainer' in st.session_state:
    st.subheader("Predict Fit Coefficients")

    include = st.session_state.get('include', [])
    comp_inputs = {}

    st.write("Enter composition fractions for selected components:")
    for comp in include:
        comp_inputs[comp] = st.number_input(
            label=f"Fraction of {comp}",
            min_value=0.0, max_value=1.0, value=0.0, format="%.3f"
        )

    if st.button("Predict"):
        trainer   = st.session_state.trainer
        model_key = st.session_state.model_key
        preds     = {}

        if model_key == "sklearn":
            # ——————————— sklearn + embedder branch ———————————
            # 1) Build and align features
            cols  = trainer.composition_df.columns
            df_in = pd.DataFrame([comp_inputs]).reindex(columns=cols, fill_value=0.0)
            X_poly = trainer.poly.transform(df_in)

            # 2) For each target, scale → embed → predict
            for target, model in trainer.best_models.items():
                # 2a) scale
                scaler    = trainer.scalers[target]
                X_scaled  = scaler.transform(X_poly)

                # 2b) embed (if an embedder was fitted)
                embedder  = trainer.embedders.get(target, None)
                if embedder is not None:
                    X_used = embedder.transform(X_scaled)
                else:
                    X_used = X_scaled

                # 2c) predict
                val = model.predict(X_used)[0]
                # preserve non‐negativity for density fits
                if "a" in target:
                    val = max(val, 1e-10)
                preds[target] = val

        else:
            # —————— resnet / kan / snn branch (unchanged) ——————
            preds = trainer.predict(comp_inputs)

        # 3) Store and display
        st.session_state.last_preds = preds
        st.subheader("Predicted Fit Coefficients")
        st.json(preds)

else:
    st.info("Train a model to enable predictions.")


# ─────────────────────────────────────────────────────────────────────────────
# Part 5: Temperature-Range Plotting
# ─────────────────────────────────────────────────────────────────────────────
if 'last_preds' in st.session_state:
    st.subheader("Plot Properties vs Temperature Range")

    Tmin, Tmax = st.slider(
        "Temperature range (K)",
        min_value=300,
        max_value=1500,
        value=(300, 900),
        step=10
    )

    if st.button("Plot Range"):
        coeffs = st.session_state.last_preds
        # Map property keys to labels and equations
        prop_labels = {
            "rho": "Density ρ (kg/m³)",
            "muA": "Viscosity μ (Arrhenius)",
            "k":   "Thermal Conductivity k (W/m·K)",
            "cp":  "Heat Capacity cₚ (J/kg·K)"
        }
        # Generate temperatures
        Ts = np.linspace(Tmin, Tmax, 100)

        for name in prop_labels:
            # Compute values and equation string
            ys = []
            for T in Ts:
                props = derive_properties(coeffs, T)
                ys.append(props.get(name, 0.0))

            # Build equation legend
            a_rho  = coeffs.get("rho_a",  0.0)
            b_rho  = coeffs.get("rho_b",  0.0)
            a_mu1  = coeffs.get("mu1_a",  0.0)
            b_mu1  = coeffs.get("mu1_b",  0.0)
            a_k    = coeffs.get("k_a",    0.0)
            b_k    = coeffs.get("k_b",    0.0)
            a_cp   = coeffs.get("cp_a",   0.0)
            b_cp   = coeffs.get("cp_b",   0.0)
            c_cp   = coeffs.get("cp_c",   0.0)

            if name == "rho":
                eq = f"ρ = {a_rho:.2e} - {b_rho:.2e}·T"
            elif name == "muA":
                eq = f"μ = {a_mu1:.2e}·exp({b_mu1:.2e}/(R·T))"
            elif name == "k":
                eq = f"k = {a_k:.2e} + {b_k:.2e}·T"
            else:  # cp
                eq = f"cₚ = {a_cp:.2e} + {b_cp:.2e}·T + {c_cp:.2e}/T²"

            # Plot
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(Ts, ys, linewidth=2, marker='o', markersize=4)
            if name == 'muA':
                ax.set_yscale('log')
            ax.set_xlabel('Temperature (K)', fontsize=12)
            ax.set_ylabel(prop_labels[name], fontsize=12)
            ax.set_title(f"{prop_labels[name]} vs Temperature", fontsize=14)
            ax.legend([eq], fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
else:
    st.info("Predict coefficients to enable temperature‐range plotting.")

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm

# ─────────────────────────────────────────────────────────────────────────────
# Part 6: Composition‐Evolution Plotting
# ─────────────────────────────────────────────────────────────────────────────
if 'last_preds' in st.session_state:
    st.subheader("Composition Evolution vs Fraction")

    comps = st.multiselect(
        "Choose components to plot:",
        options=st.session_state.get('include', []),
        default=st.session_state.get('include', [])[:3]
    )
    T = st.slider("Fixed temperature (K)", 300, 1500, 900)

    if st.button("Plot vs Composition"):
        trainer   = st.session_state.trainer
        model_key = st.session_state.model_key
        grid      = np.linspace(0, 1, 50)

        def predict_coeffs(comp_dict):
            # ── sklearn branch ─────────────────────────────────────────
            if model_key == "sklearn":
                # 1) align to the same columns used during training
                df_in = pd.DataFrame([comp_dict]).reindex(
                    columns=trainer.composition_df.columns,
                    fill_value=0.0
                )
                # 2) polynomial expansion
                X_poly = trainer.poly.transform(df_in)
                out = {}
                # 3) for each target: scale → embed → predict
                for tgt, mdl in trainer.best_models.items():
                    # scale
                    X_scaled = trainer.scalers[tgt].transform(X_poly)
                    # embed (if embedder exists for this tgt)
                    emb = getattr(trainer, "embedders", {}).get(tgt, None)
                    X_used = emb.transform(X_scaled) if emb else X_scaled
                    # predict
                    p = mdl.predict(X_used)[0]
                    if "a" in tgt:
                        p = max(p, 1e-10)
                    out[tgt] = p
                return out

            # ── neural‐net branch ───────────────────────────────────────
            # resnet / kan / snn all implement predict(dict) → dict
            return trainer.predict(comp_dict)


        # ─── 1‐component sweep ─────────────────────────────────────────────
        if len(comps) == 1:
            comp = comps[0]
            data = []
            for f in grid:
                coeffs = predict_coeffs({comp: f})
                props  = derive_properties(coeffs, T)
                data.append(props.get("rho", 0.0))  # e.g. density

            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(grid, data, '-o', linewidth=2, markersize=4)
            ax.set_xlabel(f"{comp} Fraction")
            ax.set_ylabel("Density ρ")
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

        # ─── 2‐component sweep ─────────────────────────────────────────────
        elif len(comps) == 2:
            c1, c2 = comps
            data = []
            for f in grid:
                coeffs = predict_coeffs({c1: f, c2: 1-f})
                props  = derive_properties(coeffs, T)
                data.append(props.get("rho", 0.0))

            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(grid, data, '-o', linewidth=2, markersize=4)
            ax.set_xlabel(f"{c1} Fraction")
            ax.set_ylabel("Density ρ")
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

        # ─── 3‐component (ternary) ─────────────────────────────────────────
        else:
            labels = comps[:3]
            n_coarse = 20

            for name in ['rho','muA','k','cp']:
                xs, ys, cs = [], [], []
                # coarse
                for i in range(n_coarse+1):
                    for j in range(n_coarse+1-i):
                        kf = n_coarse - i - j
                        f1, f2, f3 = i/n_coarse, j/n_coarse, kf/n_coarse
                        coeffs = predict_coeffs({
                            labels[0]: f1,
                            labels[1]: f2,
                            labels[2]: f3
                        })
                        val = derive_properties(coeffs, T).get(name)
                        if val is not None:
                            xs.append(0.5*(2*f2 + f3))
                            ys.append((np.sqrt(3)/2)*f3)
                            cs.append(val)

                if not xs:
                    continue

                pts   = np.vstack((xs, ys)).T
                interp = LinearNDInterpolator(pts, cs)

                # fine
                n_fine = 100
                xf, yf = [], []
                for i in range(n_fine+1):
                    for j in range(n_fine+1-i):
                        kf = n_fine - i - j
                        f1, f2, f3 = i/n_fine, j/n_fine, kf/n_fine
                        xf.append(0.5*(2*f2 + f3))
                        yf.append((np.sqrt(3)/2)*f3)
                xf = np.array(xf); yf = np.array(yf)
                zf = interp(xf, yf)
                mask = ~np.isnan(zf)
                xf, yf, zf = xf[mask], yf[mask], zf[mask]

                tri = mtri.Triangulation(xf, yf)
                fig, ax = plt.subplots(figsize=(6,6))
                if name.startswith('mu'):
                    cf = ax.tricontourf(tri, np.abs(zf), levels=20,
                                        cmap='viridis', norm=LogNorm(), alpha=0.9)
                else:
                    cf = ax.tricontourf(tri, np.abs(zf), levels=20,
                                        cmap='viridis', alpha=0.9)
                plt.colorbar(cf, ax=ax, label=name)

                # border & labels
                verts_x = [0,1,0.5,0]; verts_y = [0,0,np.sqrt(3)/2,0]
                ax.plot(verts_x, verts_y, 'k-', lw=1)
                ax.text(0, -0.05, labels[0], ha='center', va='top')
                ax.text(1, -0.05, labels[1], ha='center', va='top')
                ax.text(0.5, np.sqrt(3)/2+0.03,
                        labels[2], ha='center', va='bottom')

                ax.set_title(f"{name} ternary at {T} K")
                ax.axis('off')
                st.pyplot(fig)
else:
    st.info("Predict coefficients and filter data to enable composition‐evolution plotting.")
