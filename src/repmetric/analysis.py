"""
Analysis tools for CPED and geometric data analysis.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.stats import linregress
from scipy.optimize import minimize
from sklearn.manifold import MDS
from sklearn.base import BaseEstimator, TransformerMixin

def sliding_windows(s: str, k: int, step: int = 1) -> Tuple[List[str], List[int]]:
    """Extracts sliding windows from a string.

    Args:
        s: The input string.
        k: The window size.
        step: The step size.

    Returns:
        A tuple containing a list of substrings and a list of start indices.
    """
    subs = []
    starts = []
    i = 0
    while i + k <= len(s):
        subs.append(s[i : i + k])
        starts.append(i)
        i += step
    return subs, starts


def calculate_maximal_bandwidth(
    mat: np.ndarray, step: int, r2_thresh: float = 0.99
) -> Tuple[int, float, float]:
    """Calculates the maximal bandwidth where the distance increases linearly.

    Args:
        mat: The distance matrix.
        step: The step size used in sliding windows.
        r2_thresh: The R-squared threshold for linearity.

    Returns:
        A tuple containing:
        - max_bw: The maximal bandwidth.
        - fit_band_max: The fitted value at max_bw.
        - off_band_mean: The mean distance outside the bandwidth.
    """
    rows, cols = mat.shape
    i_idx, j_idx = np.indices((rows, cols))
    dists = np.abs(i_idx - j_idx).flatten()
    values = mat.flatten()

    bandwidths = range(2, 3 * step)

    max_bw = 1
    slope = 0.0
    
    # Pre-calculate masks to avoid repeated work if possible, 
    # but loop is fine for typical sizes.
    for k in bandwidths:
        mask = dists < k
        if np.sum(mask) < 2:
            continue

        s, _, r_value, _, _ = linregress(
            dists[mask],
            values[mask]
        )
        r_squared = r_value**2

        if r_squared >= r2_thresh:
            max_bw = k
            slope = s
        else:
            break

    off_band_mean = np.mean(mat[np.abs(i_idx - j_idx) > max_bw])
    return max_bw, float(slope * max_bw), float(off_band_mean)


class MDS_OOS(BaseEstimator, TransformerMixin):
    """Multidimensional Scaling with Out-of-Sample extension.

    Uses sklearn.manifold.MDS for fitting the reference data, and
    scipy.optimize.minimize to project new points by minimizing stress.
    """

    def __init__(
        self,
        n_components: int = 2,
        metric: bool = True,
        n_init: int = 4,
        max_iter: int = 300,
        verbose: int = 0,
        eps: float = 1e-3,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
        dissimilarity: str = "precomputed",
    ):
        self.n_components = n_components
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.verbose = verbose
        self.eps = eps
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.dissimilarity = dissimilarity
        
        self.mds_ = MDS(
            n_components=n_components,
            metric=metric,
            n_init=n_init,
            max_iter=max_iter,
            verbose=verbose,
            eps=eps,
            n_jobs=n_jobs,
            random_state=random_state,
            dissimilarity=dissimilarity,
        )
        self.embedding_ = None
        self._X_fit = None

    def fit(self, X: np.ndarray, y=None):
        """Fit the MDS model to X.

        Args:
            X: Distance matrix (if dissimilarity='precomputed') or feature matrix.
        """
        self.embedding_ = self.mds_.fit_transform(X)
        self._X_fit = X # Keep reference if needed, mainly we need embedding_
        return self

    def fit_transform(self, X: np.ndarray, y=None):
        return self.fit(X).embedding_

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        """Project new points into the MDS space.

        Args:
            X_new: Distance matrix between new points and fitted points.
                   Shape (n_new_samples, n_fitted_samples).
                   Assumes dissimilarity='precomputed'.

        Returns:
            Embedding of new points. Shape (n_new_samples, n_components).
        """
        if self.embedding_ is None:
            raise RuntimeError("MDS_OOS must be fitted before transform.")

        n_new = X_new.shape[0]
        n_fitted = self.embedding_.shape[0]

        if X_new.shape[1] != n_fitted:
            raise ValueError(
                f"X_new has {X_new.shape[1]} features, expected {n_fitted}."
            )

        X_embedded = np.zeros((n_new, self.n_components))

        for i in range(n_new):
            # Objective function: Stress for a single point
            # sum ( dist(x, y_j) - delta_j )^2
            # y_j are fixed landmarks (self.embedding_)
            # delta_j are input distances (X_new[i])
            
            d_true = X_new[i]

            def stress(x):
                # Calculate distances from x to all landmarks
                d_est = np.linalg.norm(self.embedding_ - x, axis=1)
                return np.sum((d_est - d_true)**2)

            # Initial guess: centroid of landmarks weighted by inverse distance?
            # Or just mean of landmarks.
            # Or closest landmark.
            closest_idx = np.argmin(d_true)
            x0 = self.embedding_[closest_idx]

            res = minimize(stress, x0, method='BFGS')
            X_embedded[i] = res.x

        return X_embedded
