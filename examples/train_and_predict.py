import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Ensure local module path is visible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processing_saltdblean.processor import SALTDBLEANProcessor
from processing_saltdblean.trainer import AIModelTrainer

# Step 1: Load and preprocess the data
processor = SALTDBLEANProcessor.from_csv('../data/saltdblean_processed.csv')
processor.df.columns = processor.df.columns.str.strip()  # clean column names

# Add normalized composition column
processor.df['Composition'] = processor.df.apply(processor.compute_composition, axis=1)

# # Step 2: Train models
trainer = AIModelTrainer(processor.df)
trainer.train_all()

# Step 3: Make prediction
example_composition = {'Na': 0.5, 'Cl': 0.5}
predicted_props = trainer.predict(example_composition)

# Step 4: Display results
print("\nPredicted Properties for example composition:")
for k, v in predicted_props.items():
    print(f"  {k}: {v:.4f}")

# Step 5: Show evaluation metrics
print("\nModel Performance Summary (R2 and MSE):")
metrics = trainer.get_metrics()
for target, scores in metrics.items():
    print(f"\nTarget: {target}")
    for model_name, result in scores.items():
        print(f"  {model_name}: R2 = {result['R2']:.4f}, MSE = {result['MSE']:.4f}")

# Step 6: Plot R2 scores
print("\nPlotting R2 scores")
plot_dir = "sklearn_prediction_plots"
os.makedirs(plot_dir, exist_ok=True)

for target, model_scores in metrics.items():

    model_names = list(model_scores.keys())
    r2_scores = [model_scores[m]['R2'] for m in model_names]

    plt.figure(figsize=(10, 5))
    plt.bar(model_names, r2_scores)
    plt.title(f"R² Scores for {target}")
    plt.ylabel("R²")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"r2_{target}.png"))
    plt.close()

# Step 7: Plot Actual vs Predicted for best models
print("\nPlot actual vs. predicted for best models coefficients")
from sklearn.metrics import r2_score

for target in trainer.present_target_columns:

    model_path = os.path.join("..", "data", "trained_models", f"{target}.joblib")
    scaler_path = os.path.join("..", "data", "trained_models", f"{target}_scaler.joblib")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        continue

    df_target = processor.df.dropna(subset=[target])
    X = pd.json_normalize(df_target['Composition']).fillna(0)
    X_poly = trainer.poly.transform(X)
    X_scaled = trainer.scalers[target].transform(X_poly)
    y_actual = pd.to_numeric(df_target[target], errors='coerce')

    model = trainer.best_models[target]
    y_pred = model.predict(X_scaled)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_actual, y_pred, alpha=0.7)
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--')
    plt.title(f"Actual vs Predicted: {target}")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"actual_vs_predicted_{target}.png"))
    plt.close()

# Step 8: Compute actual vs predicted thermophysical properties
print("\nPlot actual vs. predicted for best model physical properties")

temperature = 900  # Kelvin
os.makedirs(plot_dir, exist_ok=True)

# Clean up all coefficient columns to ensure they are numeric
coeff_cols = ['rho_a', 'rho_b', 'mu1_a', 'mu1_b',
              'mu2_a', 'mu2_b', 'mu2_c',
              'k_a', 'k_b', 'cp_a', 'cp_b', 'cp_c', 'cp_d']

for col in coeff_cols:
    if col in processor.df.columns:
        processor.df[col] = pd.to_numeric(processor.df[col], errors='coerce')

# Replace NaNs with 0.0
processor.df.fillna(0.0, inplace=True)

# Compute actual thermophysical properties
processor.df['Actual Properties'] = processor.df.apply(
    lambda row: processor.compute_actual_properties(row, temperature), axis=1
)

# Predict coefficients and compute predicted thermophysical properties
processor.df['Predicted Coeffs'] = processor.df['Composition'].apply(trainer.predict)
processor.df['Predicted Properties'] = processor.df['Predicted Coeffs'].apply(
    lambda pred: processor.compute_actual_properties_from_predictions(pred, temperature)
)

# Define which properties to compare
properties_to_compare = [
    "Density", "Viscosity A", "Thermal Conductivity", "Heat Capacity of Liquid"
]

# Plot comparisons
for prop in properties_to_compare:
    actual_vals = []
    predicted_vals = []
    skipped = 0

    for _, row in processor.df.iterrows():
        actual = row['Actual Properties'].get(prop)
        predicted = row['Predicted Properties'].get(prop)

        if actual is None or predicted is None or actual < 1e-6:
            skipped += 1
            continue

        actual_vals.append(actual)
        predicted_vals.append(predicted)

    if actual_vals and predicted_vals:
        plt.figure(figsize=(6, 6))
        plt.scatter(actual_vals, predicted_vals, alpha=0.7)
        plt.plot([min(actual_vals), max(actual_vals)], [min(actual_vals), max(actual_vals)], 'r--')
        plt.title(f"Actual vs Predicted {prop} at {temperature} K")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.grid(True)
        plt.tight_layout()
        fname = f"actual_vs_predicted_{prop.replace(' ', '_')}.png"
        plt.savefig(os.path.join(plot_dir, fname))
        plt.close()
        print(f"Saved: {fname} ({len(actual_vals)} points, {skipped} skipped)")
    else:
        print(f"Skipped {prop} — no valid data (all {skipped} rows filtered out).")

