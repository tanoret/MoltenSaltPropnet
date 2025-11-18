import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from sklearn.feature_extraction import FeatureHasher
from sklearn.utils.validation import check_is_fitted

class EmbeddingPreconditioner(BaseEstimator, TransformerMixin):
    """Feature transformation pipeline with multiple embedding options.

    Supported methods:
    - 'none': Identity transformation
    - 'pca': Principal Component Analysis
    - 'svd': Truncated SVD
    - 'nmf': Non-negative Matrix Factorization
    - 'feature_hashing': Feature hashing
    - 'low_variance': Variance-based feature selection
    - 'tsne': t-SNE (note: transforms refit the model)
    """

    def __init__(self, method: str = 'none', n_components: int = None):
        self.method = method
        self.n_components = n_components
        self.feature_names_ = None
        self.embedder = None
        self.selected_idx_ = None

    def fit(self, X, y=None):
        """Fit the transformer to data."""
        X_df = self._validate_and_convert(X)
        self.feature_names_ = X_df.columns.tolist()

        if self.method == 'none':
            pass

        elif self.method == 'feature_hashing':
            self.embedder = FeatureHasher(
                n_features=self.n_components,
                input_type='dict',
                alternate_sign=False
            )

        elif self.method == 'tsne':
            from sklearn.manifold import TSNE
            self.embedder = TSNE(
                n_components=self.n_components,
                method='barnes_hut' if self.n_components <= 3 else 'exact',
                init='random',
                random_state=0
            )

        elif self.method == 'nmf':
            X_nn = np.clip(X_df.values, 0, None)
            self.embedder = NMF(
                n_components=self.n_components,
                init='nndsvd',
                max_iter=500,
                random_state=42
            )
            self.embedder.fit(X_nn)

        elif self.method == 'low_variance':
            variances = X_df.var(axis=0).values
            self.selected_idx_ = np.argsort(variances)[-self.n_components:][::-1]

        elif self.method == 'pca':
            self.embedder = PCA(n_components=self.n_components)
            self.embedder.fit(X_df.values)

        elif self.method == 'svd':
            self.embedder = TruncatedSVD(n_components=self.n_components)
            self.embedder.fit(X_df.values)

        else:
            raise ValueError(f"Unsupported method: {self.method}")

        return self

    def transform(self, X, y=None):
        """Transform data using the fitted transformer."""
        check_is_fitted(self)
        X_df = self._validate_and_convert(X)

        if self.method == 'none':
            return self._to_float32(X_df.values)

        elif self.method == 'feature_hashing':
            dicts = X_df.apply(
                lambda row: {str(c): float(v) for c, v in row.items() if v != 0},
                axis=1
            ).tolist()
            return self._to_float32(self.embedder.transform(dicts).toarray())

        elif self.method == 'tsne':
            # Warning: t-SNE will refit on transform!
            return self._to_float32(self.embedder.fit_transform(X_df.values))

        elif self.method == 'nmf':
            arr = np.clip(X_df.values, 0, None)
            return self._to_float32(self.embedder.transform(arr))

        elif self.method == 'low_variance':
            return self._to_float32(X_df.values[:, self.selected_idx_])

        elif self.method in ('pca', 'svd'):
            return self._to_float32(self.embedder.transform(X_df.values))

        raise ValueError(f"Unsupported method: {self.method}")

    def _validate_and_convert(self, X):
        """Validate input and convert to DataFrame."""
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if self.feature_names_:
            return pd.DataFrame(X, columns=self.feature_names_)
        return pd.DataFrame(X)

    @staticmethod
    def _to_float32(arr):
        """Ensure output array is float32."""
        return arr.astype(np.float32) if arr.dtype != np.float32 else arr
