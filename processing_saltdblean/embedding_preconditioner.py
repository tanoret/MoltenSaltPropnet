import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from sklearn.feature_extraction import FeatureHasher
from sklearn.utils.validation import check_is_fitted


class EmbeddingPreconditioner(BaseEstimator, TransformerMixin):
    """
    Feature transformation pipeline with multiple embedding options.

    Supported methods:
    - 'none': Identity transform, returns X unchanged (float32).
    - 'pca': Principal Component Analysis
    - 'svd': Truncated SVD
    - 'nmf': Non-negative Matrix Factorization
    - 'feature_hashing': Feature hashing
    - 'low_variance': Variance-based feature selection
    - 'tsne': t-SNE (refits on transform)
    """

    def __init__(self, method: str = 'none', n_components: int = None):
        self.method = method
        self.n_components = n_components
        self.feature_names_ = None
        self.embedder = None
        self.selected_idx_ = None

    # -------------------------------------------------------------
    # FIXED: "none" should not store feature names or modify X
    # -------------------------------------------------------------
    def fit(self, X, y=None):

        # Identity transform: do not store feature_names, do nothing
        if self.method == 'none':
            self.feature_names_ = None
            return self

        # All other methods use DataFrame conversion
        X_df = self._validate_and_convert(X)
        self.feature_names_ = X_df.columns.tolist()

        # PCA
        if self.method == 'pca':
            self.embedder = PCA(n_components=self.n_components)
            self.embedder.fit(X_df.values)

        # SVD
        elif self.method == 'svd':
            self.embedder = TruncatedSVD(n_components=self.n_components)
            self.embedder.fit(X_df.values)

        # NMF
        elif self.method == 'nmf':
            X_nn = np.clip(X_df.values, 0, None)
            self.embedder = NMF(
                n_components=self.n_components,
                init='nndsvd',
                max_iter=500,
                random_state=42
            )
            self.embedder.fit(X_nn)

        # Feature hashing (dict input)
        elif self.method == 'feature_hashing':
            self.embedder = FeatureHasher(
                n_features=self.n_components,
                input_type='dict',
                alternate_sign=False
            )

        # Variance-based selection
        elif self.method == 'low_variance':
            variances = X_df.var(axis=0).values
            self.selected_idx_ = np.argsort(variances)[-self.n_components:][::-1]

        # t-SNE (fit during transform)
        elif self.method == 'tsne':
            from sklearn.manifold import TSNE
            self.embedder = TSNE(
                n_components=self.n_components,
                method='barnes_hut' if self.n_components <= 3 else 'exact',
                init='random',
                random_state=0
            )

        else:
            raise ValueError(f"Unsupported method: {self.method}")

        return self

    # -------------------------------------------------------------
    # FIXED: "none" returns raw numpy X (float32)
    # -------------------------------------------------------------
    def transform(self, X, y=None):
        check_is_fitted(self)

        # Identity mode: return raw matrix unchanged
        if self.method == 'none':
            return self._to_float32(np.asarray(X))

        # For all other methods convert to DataFrame with stored names
        X_df = self._validate_and_convert(X)

        # Feature hashing
        if self.method == 'feature_hashing':
            dicts = X_df.apply(
                lambda row: {str(c): float(v) for c, v in row.items() if v != 0},
                axis=1
            ).tolist()
            return self._to_float32(self.embedder.transform(dicts).toarray())

        # t-SNE (refits every transform)
        if self.method == 'tsne':
            return self._to_float32(self.embedder.fit_transform(X_df.values))

        # NMF
        if self.method == 'nmf':
            arr = np.clip(X_df.values, 0, None)
            return self._to_float32(self.embedder.transform(arr))

        # Variance selection
        if self.method == 'low_variance':
            return self._to_float32(X_df.values[:, self.selected_idx_])

        # PCA / SVD
        if self.method in ('pca', 'svd'):
            return self._to_float32(self.embedder.transform(X_df.values))

        raise ValueError(f"Unsupported method: {self.method}")

    # -------------------------------------------------------------
    # Helper utilities
    # -------------------------------------------------------------
    def _validate_and_convert(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()

        # If feature_names_ exists, enforce correct column order
        if self.feature_names_:
            return pd.DataFrame(X, columns=self.feature_names_)

        # Default: construct DataFrame without names
        return pd.DataFrame(X)

    @staticmethod
    def _to_float32(arr):
        return arr.astype(np.float32) if arr.dtype != np.float32 else arr
