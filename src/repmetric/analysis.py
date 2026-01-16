"""
Analysis tools for CPED and geometric data analysis.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.stats import linregress
from scipy.optimize import minimize
from scipy.sparse.csgraph import dijkstra
from sklearn.manifold import MDS
from sklearn.base import BaseEstimator, TransformerMixin
from repmetric.api import edit_distance
import ot.gromov


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


def splength(mat: np.ndarray) -> float:
    """Calculates the shortest path length from start to end of the sequence graph.

    Args:
        mat: The distance matrix.

    Returns:
        The shortest path distance from the first to the last node.
    """
    directed = not np.allclose(mat, mat.T)
    return dijkstra(
        csgraph=mat, directed=directed, indices=0, return_predecessors=False
    )[-1]


def augment_dataset(
    target_sequences: List[Tuple[str, str]], augment_a: int, augment_b: int
) -> List[Tuple[str, str]]:
    """Augments the dataset using geodesic paths.

    For each sequence, computes the CPED geodesic from empty string,
    and extracts intermediate strings at specified intervals.
    """
    augmented = []
    for gene_id, sequence in target_sequences:
        # We assume sequence is just the string.
        # edit_distance returns (dist, geodesic) when geodesic=True
        res = edit_distance("", sequence, distance_type="cped", geodesic=True)
        if isinstance(res, tuple):
            _, geodesic = res
        else:
            raise ValueError("Expected tuple from edit_distance with geodesic=True")

        route = geodesic.get_path_strings()
        n = len(route) - 1

        # Avoid division by zero if augment_b is 0, though unlikely
        if augment_b == 0:
            continue

        augmented.extend(
            [
                (f"{gene_id.strip()}_{k}_of_{augment_b}", route[(k * n) // augment_b])
                for k in range(augment_a, augment_b + 1)
            ]
        )
    return augmented


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

        s, _, r_value, _, _ = linregress(dists[mask], values[mask])
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
        self.embedding_: Optional[np.ndarray] = None
        self._X_fit: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y=None):
        """Fit the MDS model to X.

        Args:
            X: Distance matrix (if dissimilarity='precomputed') or feature matrix.
        """
        self.embedding_ = self.mds_.fit_transform(X)
        self._X_fit = X  # Keep reference if needed, mainly we need embedding_
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
                return np.sum((d_est - d_true) ** 2)

            # Initial guess: centroid of landmarks weighted by inverse distance?
            # Or just mean of landmarks.
            # Or closest landmark.
            closest_idx = np.argmin(d_true)
            x0 = self.embedding_[closest_idx]

            res = minimize(stress, x0, method="BFGS")
            X_embedded[i] = res.x

        return X_embedded


def compute_gw_distance(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    loss_fun: str = "square_loss",
    max_iter: int = 10000,
    tol_rel: float = 1e-6,
    tol_abs: float = 1e-6,
    symmetric: bool = True,
) -> float:
    """Calculates the Gromov-Wasserstein distance between two distance matrices.

    Args:
        matrix_a: First distance matrix (n x n).
        matrix_b: Second distance matrix (m x m).
        loss_fun: Loss function used for the GW distance.
        max_iter: Maximum number of iterations.
        tol_rel: Relative tolerance for convergence.
        tol_abs: Absolute tolerance for convergence.
        symmetric: Whether the matrices are symmetric.

    Returns:
        The Gromov-Wasserstein distance.
    """
    n_a = matrix_a.shape[0]
    n_b = matrix_b.shape[0]
    p_a = np.ones(n_a) / n_a
    p_b = np.ones(n_b) / n_b

    gw_config = dict(
        armijo=True,
        maxiter=max_iter,
        tol_rel=tol_rel,
        tol_abs=tol_abs,
        loss_fun=loss_fun,
    )

    return ot.gromov.gromov_wasserstein2(
        matrix_a, matrix_b, p_a, p_b, symmetric=symmetric, **gw_config
    )
