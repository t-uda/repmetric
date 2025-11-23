import numpy as np
from repmetric.analysis import sliding_windows, MDS_OOS


def test_sliding_windows():
    s = "ABCDEFG"
    k = 3
    step = 2
    subs, starts = sliding_windows(s, k, step)

    assert subs == ["ABC", "CDE", "EFG"]
    assert starts == [0, 2, 4]

    s = "ABCDEF"
    subs, starts = sliding_windows(s, k, step)
    assert subs == ["ABC", "CDE"]
    assert starts == [0, 2]


def test_mds_oos_synthetic():
    # Generate points on a circle
    n_samples = 20
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    radius = 10.0
    X_true = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

    # Compute Euclidean distance matrix
    from scipy.spatial.distance import pdist, squareform

    D = squareform(pdist(X_true))

    # Fit MDS
    mds = MDS_OOS(n_components=2, random_state=42, dissimilarity="precomputed")
    X_embedded = mds.fit_transform(D)

    # New point (also on circle, between existing points)
    new_angle = angles[0] + 0.1
    new_point = np.array([[radius * np.cos(new_angle), radius * np.sin(new_angle)]])

    # Distances from new point to existing points
    d_new = np.linalg.norm(X_true - new_point, axis=1)

    # Transform
    # Input shape for transform is (n_new, n_fitted)
    X_new_embedded = mds.transform(d_new.reshape(1, -1))

    # Check if the reconstruction is good
    # We check if the distances in embedding space match the input distances
    d_embedded = np.linalg.norm(X_embedded - X_new_embedded, axis=1)

    # Stress/Error
    error = np.mean((d_embedded - d_new) ** 2)

    # MDS is not perfect, but error should be small for perfect geometric data
    assert error < 1e-1, f"MDS OOS projection error too high: {error}"

    # Check shape
    assert X_new_embedded.shape == (1, 2)
