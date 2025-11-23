"""
Plotting tools for CPED analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import networkx as nx
from .analysis import calculate_maximal_bandwidth

def plot_network(mat: np.ndarray, threshold: float):
    """Plots a network graph from the distance matrix.

    Args:
        mat: The distance matrix.
        threshold: The threshold for edge creation.

    Returns:
        The NetworkX graph object.
    """
    num_windows = mat.shape[0]
    i, j = np.indices((num_windows, num_windows))
    mask = np.logical_and(0 < mat, mat < threshold)
    weighted_edges = np.array([*zip(i[mask], j[mask], mat[mask])])
    G = nx.DiGraph()
    G.add_nodes_from(np.arange(num_windows))
    G.add_weighted_edges_from(weighted_edges)
    
    # weights is not the same with unrescaled_dist in G's ordering
    weights = np.array([data['weight'] for u, v, data in G.edges(data=True)])
    
    # Avoid division by zero or log of zero/negative if weights are weird
    # The original code: edge_color = 1 - np.exp(1 - weights)
    # Assuming weights are distances >= 1 usually?
    # If weights are small, exp(1-small) is close to e.
    # Let's keep original logic but be careful.
    edge_color = 1 - np.exp(1 - weights) 

    # Use spring_layout as a standard fallback for forceatlas2
    pos = nx.spring_layout(G, weight='weight', seed=42)

    plt.figure(figsize=(8, 4), dpi=100)
    nx.draw(
        G, pos,
        with_labels=False, node_size=20, width=0.2,
        node_color=np.linspace(0, 1, num_windows), alpha=0.7,
        edge_color=edge_color,
        cmap='rainbow', edge_cmap=colormaps['Greys']
    )
    plt.show()
    return G


def plot_distance_matrix(
    M: np.ndarray, step: int = 1, r2_thresh: float = 0.95, show_network: bool = True
):
    """Plots the distance matrix as a heatmap and optionally the network.

    Args:
        M: The distance matrix.
        step: The step size.
        r2_thresh: R-squared threshold for bandwidth calculation.
        show_network: Whether to show the network plot.
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
    im = ax0.matshow(M, cmap="viridis")
    ax0.set_xlabel("Window index j")
    ax0.set_ylabel("Window index i")
    plt.colorbar(im, ax=ax0, label="Distance")
    
    max_bandwidth, fit_band_max, offband_mean = calculate_maximal_bandwidth(M, step, r2_thresh)
    print(f"{max_bandwidth=} {fit_band_max=} {offband_mean=}")
    
    thresh_min, thresh_max = sorted([fit_band_max, offband_mean])
    B = np.where(M < thresh_min, 1, np.where(M > thresh_max, 0, 0.5))
    
    ax1.matshow(B, cmap="Greys")
    ax1.set_xlabel("Window index j")
    ax1.set_ylabel("Window index i")
    plt.show()
    
    if show_network:
        plot_network(M, thresh_min)
