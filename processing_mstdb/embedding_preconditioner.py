import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD, NMF, KernelPCA, FastICA
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap
from sklearn.feature_extraction import FeatureHasher
from sklearn.utils.validation import check_is_fitted

class EmbeddingPreconditioner(BaseEstimator, TransformerMixin):
    """
    Feature transformation pipeline with multiple embedding options.
    Supported methods:
    - 'none': Identity transformation
    - 'pca': Principal Component Analysis
    - 'svd': Truncated SVD
    - 'nmf': Non-negative Matrix Factorization
    - 'feature_hashing': Feature hashing
    - 'low_variance': Variance-based feature selection
    - 'tsne': t-SNE (note: transforms refit the model)
    - 'kernel_pca': Kernel Principal Component Analysis
    - 'ica': Independent Component Analysis
    - 'lle': Locally Linear Embedding
    - 'isomap': Isomap
    """
    def __init__(self, method: str = 'none', n_components: int = None, kernel: str = 'rbf'):
        self.method = method
        self.n_components = n_components
        self.kernel = kernel
        self.feature_names_ = None
        self.embedder = None
        self.selected_idx_ = None

    def _validate_and_convert(self, X):
        """Validate input and convert to DataFrame."""
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, np.ndarray):
            if self.feature_names_ and len(self.feature_names_) != X.shape[1]:
                raise ValueError(f"Expected {len(self.feature_names_)} features, got {X.shape[1]}")
            columns = self.feature_names_ if self.feature_names_ else None
            return pd.DataFrame(X, columns=columns)
        raise ValueError("Input must be DataFrame or array-like.")

    @staticmethod
    def _to_float32(arr):
        """Ensure output array is float32."""
        return arr.astype(np.float32) if arr.dtype != np.float32 else arr

    def fit(self, X, y=None):
        """Fit the transformer to data."""
        X_df = self._validate_and_convert(X)
        self.feature_names_ = X_df.columns.tolist()

        if self.n_components is not None and self.n_components > X_df.shape[1]:
            raise ValueError(f"n_components ({self.n_components}) cannot exceed number of features ({X_df.shape[1]})")

        if self.method == 'none':
            pass
        elif self.method == 'feature_hashing':
            self.embedder = FeatureHasher(
                n_features=self.n_components or X_df.shape[1],
                input_type='dict',
                alternate_sign=False
            )
        elif self.method == 'tsne':
            self.embedder = TSNE(
                n_components=self.n_components or 2,
                method='barnes_hut' if (self.n_components or 2) <= 3 else 'exact',
                init='random',
                random_state=0
            )
        elif self.method == 'nmf':
            X_nn = np.clip(X_df.values, 0, None)
            self.embedder = NMF(
                n_components=self.n_components or X_df.shape[1],
                init='nndsvd',
                max_iter=500,
                random_state=42
            )
            self.embedder.fit(X_nn)
        elif self.method == 'low_variance':
            variances = X_df.var(axis=0).values
            n_components = self.n_components or X_df.shape[1]
            if len(variances) < n_components:
                raise ValueError(f"Cannot select {n_components} features; only {len(variances)} available")
            self.selected_idx_ = np.argsort(variances)[-n_components:][::-1]
        elif self.method == 'pca':
            self.embedder = PCA(n_components=self.n_components)
            self.embedder.fit(X_df.values)
        elif self.method == 'svd':
            self.embedder = TruncatedSVD(n_components=self.n_components or X_df.shape[1])
            self.embedder.fit(X_df.values)
        elif self.method == 'kernel_pca':
            self.embedder = KernelPCA(
                n_components=self.n_components,
                kernel=self.kernel,
                random_state=42
            )
            self.embedder.fit(X_df.values)
        elif self.method == 'ica':
            self.embedder = FastICA(
                n_components=self.n_components,
                random_state=42
            )
            self.embedder.fit(X_df.values)
        elif self.method == 'lle':
            self.embedder = LocallyLinearEmbedding(
                n_components=self.n_components or 2,
                n_neighbors=10,
                random_state=42
            )
            self.embedder.fit(X_df.values)
        elif self.method == 'isomap':
            self.embedder = Isomap(
                n_components=self.n_components or 2,
                n_neighbors=5,
                random_state=42
            )
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
        elif self.method in ('pca', 'svd', 'kernel_pca', 'ica', 'lle', 'isomap'):
            return self._to_float32(self.embedder.transform(X_df.values))
        else:
            raise ValueError(f"Unsupported method: {self.method}")




"""
#Example to test the embeddeder
X = pd.DataFrame(np.random.rand(100, 14), columns=[f"feature_{i}" for i in range(14)])

# Test PCA
embedder = EmbeddingPreconditioner(method='pca', n_components=5)
embedder.fit(X)
X_transformed = embedder.transform(X)
print(X_transformed.shape)  

# Test Kernel PCA
embedder = EmbeddingPreconditioner(method='kernel_pca', n_components=3, kernel='linear')
embedder.fit(X)
X_transformed = embedder.transform(X)
print(X_transformed.shape) 
"""