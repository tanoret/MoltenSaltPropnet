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
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from pathlib import Path

from sklearn.metrics import mean_squared_error, r2_score

def _rel_mse_pct_scalar(mse: float, y_series: pd.Series) -> float:
    """Relative MSE (% of ⟨y²⟩) given an absolute mse and the true values."""
    denom = (y_series.astype(float) ** 2).mean()
    denom = denom if np.isfinite(denom) and denom > 0 else 1e-12
    return 100.0 * mse / denom


class AIModelTrainer:
    def __init__(self, df: pd.DataFrame, composition_column: str = 'Composition',
                 embedding_method: str = 'none', embedding_params: Dict = None):
        self.df = df.copy()
        self.composition_column = composition_column
        self.composition_df = pd.json_normalize(self.df[self.composition_column]).fillna(0.0)
        self.composition_df = self.composition_df.reindex(sorted(self.composition_df.columns), axis=1)
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.poly.fit(self.composition_df)
        self.scalers = {}
        self.embedders = {}
        self.best_models = {}
        self.results = {}
        self.embedding_method = embedding_method
        self.embedding_params = embedding_params or {}
        self.target_columns = ['Melt(K)', 'Boil(K)', 'rho_a', 'rho_b', 'mu1_a', 'mu1_b',
                              'mu2_b', 'mu2_a', 'mu2_c', 'k_a', 'k_b',
                              'cp_a', 'cp_b', 'cp_c', 'cp_d']
        self.present_target_columns = [col for col in self.target_columns if col in self.df.columns]
        self.all_elements = self.composition_df.columns

    @staticmethod
    def parse_compound(c: str) -> Dict[str, int]:
        """Parse a compound into its constituent elements (e.g., 'NaCl' → {'Na': 1, 'Cl': 1})."""
        out = {}
        for el, n in re.findall(r"([A-Z][a-z]*)(\d*)", c):
            out[el] = out.get(el, 0) + int(n or "1")
        return out

    def _get_embedder(self):
        """Return the appropriate embedder based on the specified embedding method."""
        if self.embedding_method == 'pca':
            return PCA(**self.embedding_params)
        elif self.embedding_method == 'nmf':
            return NMF(**self.embedding_params)
        elif self.embedding_method == 'svd':
            return TruncatedSVD(**self.embedding_params)
        elif self.embedding_method == 'low_variance':
            return VarianceThreshold(**self.embedding_params)
        else:
            raise ValueError(f"Unsupported embedding method: {self.embedding_method}")

    def train_all(self):
        for target in self.present_target_columns:
            try:
                self.df[target] = pd.to_numeric(self.df[target], errors='coerce')
                df_target = self.df.dropna(subset=[target])
                y = df_target[target]
                X_target = self.composition_df.loc[df_target.index]
                X_poly = self.poly.transform(X_target)

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_poly)
                self.scalers[target] = scaler

                if len(y) >= 3:
                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.05, random_state=42)
                else:
                    X_train, X_test, y_train, y_test = X_scaled, X_scaled, y, y

                if self.embedding_method != 'none':
                    embedder = self._get_embedder()
                    embedder.fit(X_train)
                    X_train = embedder.transform(X_train)
                    X_test = embedder.transform(X_test)
                    self.embedders[target] = embedder
                else:
                    self.embedders[target] = None

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
                    if self.embedding_method != 'none':
                        joblib.dump(self.embedders[target], model_dir / f'{target}_embedder.joblib')

            except Exception as e:
                print(f"Error processing {target}: {e}")

    def predict(self, elemental_composition: Dict[str, float]) -> Dict[str, float]:
        # Decompose compounds into elements and sum contributions
        elements = {}
        compounds = {}

        for key, value in elemental_composition.items():
            parsed = self.parse_compound(key)
            if len(parsed) > 1:  # Compound (e.g., NaCl)
                compounds[key] = compounds.get(key, 0.0) + value
                for el, count in parsed.items():
                    elements[el] = elements.get(el, 0.0) + value * count
            else:  # Element (e.g., Na)
                el = list(parsed.keys())[0]
                elements[el] = elements.get(el, 0.0) + value

        # Combine elements and compounds, then normalize
        combined = {**elements, **compounds}
        total = sum(combined.values())
        if total <= 0:
            raise ValueError("Total composition must be positive.")
        normalized = {k: v / total for k, v in combined.items()}

        # Map to expected features (sorted to match training order)
        input_composition = {
            el: normalized.get(el, 0.0) for el in self.all_elements
        }

        # Generate predictions
        input_df = pd.DataFrame([input_composition])
        predictions = {}
        for target in self.present_target_columns:
            model_path = Path('../data/trained_models') / f'{target}.joblib'
            scaler_path = Path('../data/trained_models') / f'{target}_scaler.joblib'
            poly_path = Path('../data/trained_models') / f'{target}_poly_transformer.joblib'
            embedder_path = Path('../data/trained_models') / f'{target}_embedder.joblib'
            if model_path.exists() and scaler_path.exists() and poly_path.exists():
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                poly = joblib.load(poly_path)
                input_poly = poly.transform(input_df)
                input_scaled = scaler.transform(input_poly)
                if self.embedding_method != 'none' and embedder_path.exists():
                    embedder = joblib.load(embedder_path)
                    input_embedded = embedder.transform(input_scaled)
                else:
                    input_embedded = input_scaled
                prediction = model.predict(input_embedded)[0]
                if 'a' in target:
                    prediction = max(prediction, 1e-10)
                predictions[target] = prediction
        return predictions

    def get_metrics(self):
        return self.results

    def summary_metrics(self) -> dict:
        """Return avg. relative-MSE (%) and R², plus per-target table."""
        per_target, rels, r2s = {}, [], []

        for tgt, mdl_dict in self.results.items():
            # pick the model with the highest R²
            best = max(mdl_dict.values(), key=lambda d: d["R2"])
            mse_abs, r2 = best["MSE"], best["R2"]

            # compute relative MSE (%) w.r.t. mean squared true values
            m_rel = _rel_mse_pct_scalar(mse_abs, self.df[tgt].dropna())

            per_target[tgt] = {"MSE_pct": float(m_rel), "R2": float(r2)}
            rels.append(m_rel);  r2s.append(r2)

        return {
            "avg_mse_pct": float(np.mean(rels)) if rels else float("nan"),
            "avg_r2"     : float(np.mean(r2s))  if r2s  else float("nan"),
            "per_target" : per_target
        }
