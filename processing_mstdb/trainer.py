import os
import re
import joblib
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from pathlib import Path


class AIModelTrainer:
    def __init__(self, df: pd.DataFrame, composition_column: str = 'Composition'):
        self.df = df.copy()
        self.composition_column = composition_column
        self.composition_df = pd.json_normalize(self.df[self.composition_column]).fillna(0.0)
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.poly.fit(self.composition_df)
        self.scalers = {}
        self.best_models = {}
        self.results = {}
        self.target_columns = ['Melt(K)', 'Boil(K)', 'rho_a', 'rho_b', 'mu1_a', 'mu1_b',
                               'mu2_b', 'mu2_a', 'mu2_c', 'k_a', 'k_b',
                               'cp_a', 'cp_b', 'cp_c', 'cp_d']
        self.present_target_columns = [col for col in self.target_columns if col in self.df.columns]
        self.all_elements = self.composition_df.columns

    def train_all(self):
        for target in self.present_target_columns:
            try:
                self.df[target] = pd.to_numeric(self.df[target], errors='coerce')
                df_target = self.df.dropna(subset=[target])
                y = df_target[target]
                X_target = self.composition_df.loc[df_target.index]
                X_poly = self.poly.transform(X_target)

                scaler = StandardScaler()
                X = scaler.fit_transform(X_poly)
                self.scalers[target] = scaler

                if len(y) >= 3:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
                else:
                    X_train, X_test, y_train, y_test = X, X, y, y

                models = {
                    'Linear Regression': LinearRegression(),
                    'Ridge Regression': GridSearchCV(Ridge(), param_grid={'alpha': [0.1, 1.0, 10.0]}, cv=5),
                    'Lasso Regression': GridSearchCV(Lasso(), param_grid={'alpha': [0.1, 1.0, 10.0]}, cv=5),
                    'Random Forest': GridSearchCV(RandomForestRegressor(), param_grid={'n_estimators': [50, 100, 200]}, cv=5),
                    'Gradient Boosting': GridSearchCV(GradientBoostingRegressor(), param_grid={'n_estimators': [50, 100, 200]}, cv=5),
                    'SVR': GridSearchCV(SVR(), param_grid={'C': [0.1, 1.0, 10.0], 'epsilon': [0.01, 0.1, 1.0]}, cv=5),
                    'Neural Network': GridSearchCV(MLPRegressor(), param_grid={'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.0001, 0.001, 0.01]}, cv=5)
                }

                target_results = {}
                for model_name, model in models.items():
                    if len(y) >= 3:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        mse = np.abs(mean_squared_error(y_test, y_pred))
                        r2 = max(r2_score(y_test, y_pred), 0)
                        target_results[model_name] = {'MSE': mse, 'R2': r2}
                    else:
                        target_results[model_name] = {'MSE': float('inf'), 'R2': float('-inf')}

                self.results[target] = target_results

                if len(y) >= 3:
                    best_model_name = max(target_results, key=lambda k: target_results[k]['R2'])
                    best_model = models[best_model_name].best_estimator_ if isinstance(models[best_model_name], GridSearchCV) else models[best_model_name]
                    self.best_models[target] = best_model
                    model_dir = Path('../data/trained_models')
                    model_dir.mkdir(parents=True, exist_ok=True)
                    joblib.dump(best_model, model_dir / f'{target}.joblib')
                    joblib.dump(self.scalers[target], model_dir / f'{target}_scaler.joblib')
                    joblib.dump(self.poly, model_dir / f'{target}_poly_transformer.joblib')

            except Exception as e:
                print(f"Error processing {target}: {e}")

    def predict(self, elemental_composition: Dict[str, float]) -> Dict[str, float]:
        input_composition = {el: elemental_composition.get(el, 0) for el in self.all_elements}
        predictions = {}
        input_df = pd.DataFrame([input_composition])
        for target in self.present_target_columns:
            model_path = Path('../data/trained_models') / f'{target}.joblib'
            scaler_path = Path('../data/trained_models') / f'{target}_scaler.joblib'
            poly_path = Path('../data/trained_models') / f'{target}_poly_transformer.joblib'
            if model_path.exists() and scaler_path.exists() and poly_path.exists():
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                poly = joblib.load(poly_path)
                input_poly = poly.transform(input_df)
                input_scaled = scaler.transform(input_poly)
                prediction = model.predict(input_scaled)[0]
                if 'a' in target:
                    prediction = max(prediction, 1e-10)
                predictions[target] = prediction
        return predictions

    def get_metrics(self):
        return self.results
