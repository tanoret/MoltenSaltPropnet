import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from processing_mstdb.processor import MSTDBProcessor

processor = MSTDBProcessor.from_csv('/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv')
print(processor.df.head())
processor.df.columns = processor.df.columns.str.strip()  # clean column names
print(processor.df.columns)

# Compute compositions
compositions = []
for idx, row in processor.df.iterrows():
    comp = processor.compute_composition(row, composition_type='elements')
    compositions.append(comp)
processor.df['Composition'] = compositions

all_elements = sorted(processor.predefined_elements)
X_composition = np.zeros((len(processor.df), len(all_elements)))
for idx, comp in enumerate(compositions):
    for el, frac in comp.items():
        if el in all_elements:
            X_composition[idx, all_elements.index(el)] = frac

T_min, T_max = 300, 2000
T_range = np.linspace(T_min, T_max, 100).reshape(-1, 1)

property_columns = ['rho_a', 'rho_b', 'mu1_a', 'mu1_b', 'mu2_a', 'mu2_b', 'mu2_c', 'k_a', 'k_b', 'cp_a', 'cp_b']
y_properties = processor.df[property_columns].values  # Keep NaNs

# Compute mean/std ignoring NaNs
mean_y = np.nanmean(y_properties, axis=0)
std_y = np.nanstd(y_properties, axis=0) + 1e-10

# Standardize non-NaN values, set NaNs to 0 for tensor
y_standardized = np.where(np.isnan(y_properties), 0.0, (y_properties - mean_y) / std_y)

# Mask for original non-NaNs (True where non-NaN)
mask = ~np.isnan(y_properties)

mean_X, std_X = X_composition.mean(axis=0), X_composition.std(axis=0) + 1e-10
X_composition = processor.standardize_with_nan(X_composition, mean_X, std_X)

# --- Train/Validation Split (80/20 row-wise) ---
# Split X, standardized y, and mask together
indices = np.arange(len(X_composition))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_train_tensor = tf.convert_to_tensor(X_composition[train_idx], dtype=tf.float32)
X_val_tensor = tf.convert_to_tensor(X_composition[val_idx], dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_standardized[train_idx], dtype=tf.float32)
y_val_tensor = tf.convert_to_tensor(y_standardized[val_idx], dtype=tf.float32)
mask_train_tensor = tf.convert_to_tensor(mask[train_idx], dtype=tf.bool)
mask_val_tensor = tf.convert_to_tensor(mask[val_idx], dtype=tf.bool)

#  PINNModel Class 
class PINNModel(Model):
    def __init__(self, num_elements, hidden_units=[64, 64], activation='relu'):
        super(PINNModel, self).__init__()
        if activation == 'leaky_relu':
            activation_func = tf.keras.layers.LeakyReLU(alpha=0.01)
        elif activation == 'swish':
            activation_func = tf.keras.activations.swish
        elif activation == 'gelu':
            activation_func = tf.keras.activations.gelu
        else:
            activation_func = activation  # relu, tanh, elu
        self.dense_layers = [layers.Dense(units, activation=activation_func) for units in hidden_units]
        self.output_layer = layers.Dense(len(property_columns))
        self.R = 8.314

    def call(self, inputs):
        composition, temperature = inputs
        x = composition
        for layer in self.dense_layers:
            x = layer(x)
        coeffs = self.output_layer(x)
        return coeffs

    def compute_properties(self, coeffs, T):
        rho_a, rho_b, mu1_a, mu1_b, mu2_a, mu2_b, mu2_c, k_a, k_b, cp_a, cp_b = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2], coeffs[:, 3], coeffs[:, 4], coeffs[:, 5], coeffs[:, 6], coeffs[:, 7], coeffs[:, 8], coeffs[:, 9], coeffs[:, 10]
        properties = {
            'density': tf.minimum(tf.maximum(rho_a - rho_b * T, 0.0), 1000.0),
            'viscosity_A': tf.where((mu1_a > 1e-10) & (tf.abs(mu1_b) > 1e-10),
                                    tf.minimum(tf.maximum(mu1_a * tf.exp(mu1_b / (self.R * T)), 0.0), 200.0), 0.0),
            'viscosity_B': tf.where((tf.abs(mu2_a) > 1e-10) & (tf.abs(mu2_b) > 1e-10) & (tf.abs(mu2_c) > 1e-10),
                                    tf.minimum(tf.maximum(tf.pow(10.0, mu2_a + mu2_b / T + mu2_c / T**2), 0.0), 200.0), 0.0),
            'thermal_conductivity': tf.minimum(tf.maximum(k_a + k_b * T, 0.0), 100.0),
            'heat_capacity': tf.minimum(tf.maximum(cp_a + cp_b * T, 0.0), 200.0)
        }
        return properties

    def compute_derivatives(self, properties, coeffs, T):
        rho_b = coeffs[:, 1]
        mu1_a, mu1_b = coeffs[:, 2], coeffs[:, 3]
        mu2_a, mu2_b, mu2_c = coeffs[:, 4], coeffs[:, 5], coeffs[:, 6]
        k_b = coeffs[:, 8]
        cp_b = coeffs[:, 10]
        derivatives = {
            'd_density_dT': -rho_b,
            'd_viscosity_A_dT': properties['viscosity_A'] * (-mu1_b / (self.R * T**2)),
            'd_viscosity_B_dT': properties['viscosity_B'] * tf.math.log(10.0) * (-mu2_b / T**2 - 2 * mu2_c / T**3),
            'd_thermal_conductivity_dT': k_b,
            'd_heat_capacity_dT': cp_b
        }
        return derivatives

# PINN Loss Function 
def pinn_loss(model, X_comp, T, y_true, mask):
    coeffs_pred = model([X_comp, T])
    # Masked data loss (ignore NaNs/0s from original NaNs)
    squared_diff = tf.square(coeffs_pred - y_true)
    masked_squared_diff = squared_diff * tf.cast(mask, tf.float32)
    num_valid = tf.reduce_sum(tf.cast(mask, tf.float32))
    data_loss = tf.reduce_sum(masked_squared_diff) / tf.maximum(num_valid, 1e-10)
    
    properties = model.compute_properties(coeffs_pred, T)
    physics_loss = 0.0
    for prop in properties.values():
        physics_loss += tf.reduce_mean(tf.square(tf.where(prop < 0, -prop, 0.0)))
    
    derivatives = model.compute_derivatives(properties, coeffs_pred, T)
    deriv_loss = 0.0
    deriv_loss += tf.reduce_mean(tf.square(tf.maximum(derivatives['d_density_dT'], 0.0)))
    deriv_loss += tf.reduce_mean(tf.square(tf.maximum(derivatives['d_viscosity_A_dT'], 0.0)))
    deriv_loss += tf.reduce_mean(tf.square(tf.maximum(derivatives['d_viscosity_B_dT'], 0.0)))
    
    total_physics_loss = physics_loss + 0.05 * deriv_loss
    return data_loss + 0.1 * total_physics_loss

# --- Optimizer and Activation Configurations ---
optimizers = {
    'Adam': lambda: tf.keras.optimizers.Adam(learning_rate=0.001),
    'AdamW': lambda: tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
    'Adagrad': lambda: tf.keras.optimizers.Adagrad(learning_rate=0.001),
    'RMSprop': lambda: tf.keras.optimizers.RMSprop(learning_rate=0.001),
    'SGD': lambda: tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'Nadam': lambda: tf.keras.optimizers.Nadam(learning_rate=0.001)
}

activations = ['relu', 'tanh', 'swish', 'gelu', 'leaky_relu', 'elu']


results = []
epochs = 100  # Adjust as needed; higher for better training, lower for faster testing

for opt_name in optimizers:
    for act_name in activations:
        print(f"\nTraining with optimizer: {opt_name}, activation: {act_name}")
        
        # Create a new model
        model = PINNModel(num_elements=len(all_elements), activation=act_name)
        optimizer = optimizers[opt_name]()
        
        # Training step function
        @tf.function
        def train_step(X, T, y, mask):
            with tf.GradientTape() as tape:
                loss = pinn_loss(model, X, T, y, mask)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss
        
        # Training loop
        for epoch in range(epochs):
            T_batch = np.random.uniform(T_min, T_max, size=(X_train_tensor.shape[0], 1))
            T_batch_tensor = tf.convert_to_tensor(T_batch, dtype=tf.float32)
            
            train_loss = train_step(X_train_tensor, T_batch_tensor, y_train_tensor, mask_train_tensor)
            
            # Evaluate on validation set
            T_val_batch = np.random.uniform(T_min, T_max, size=(X_val_tensor.shape[0], 1))
            T_val_batch_tensor = tf.convert_to_tensor(T_val_batch, dtype=tf.float32)
            val_loss = pinn_loss(model, X_val_tensor, T_val_batch_tensor, y_val_tensor, mask_val_tensor)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Train Loss: {train_loss.numpy():.4f}, Val Loss: {val_loss.numpy():.4f}")
        
        # Record final validation loss
        results.append({
            'optimizer': opt_name,
            'activation': act_name,
            'final_val_loss': val_loss.numpy()
        })

# --- Display Results ---
results_df = pd.DataFrame(results)
print("\nFinal Results:")
print(results_df)

# Find and print the best combination (lowest validation loss)
best_result = results_df.loc[results_df['final_val_loss'].idxmin()]
print(f"\nBest Combination: Optimizer = {best_result['optimizer']}, Activation = {best_result['activation']}, Final Val Loss = {best_result['final_val_loss']:.4f}")

#print MSE for each property for the best model and r2
from sklearn.metrics import mean_squared_error, r2_score

