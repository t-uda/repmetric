"""High level API for repmetric operations."""

from __future__ import annotations

from typing import List, Literal, Union, overload

import numpy as np

from .backend import (
    CPP_AVAILABLE,
    _calculate_cped_cpp,
    _calculate_cped_distance_matrix_cpp,
    _calculate_cped_distance_matrix_py,
    _calculate_cped_distance_matrix_py_bidirectional,
    _calculate_cped_py,
    _calculate_cped_py_bidirectional,
    _calculate_levd_cpp,
    _calculate_levd_distance_matrix_cpp,
    _calculate_levd_distance_matrix_py,
    _calculate_levd_py,
)

Backend = Literal["cpp", "c++", "python"]
DistanceType = Literal["cped", "cped-bidir", "cped_bidirectional", "levd"]


def _normalize_distance_type(distance_type: str) -> str:
    """Return the canonical lowercase distance type name."""

    normalized = distance_type.lower()
    if normalized == "cped_bidirectional":
        return "cped-bidir"
    return normalized


def cped(X: str, Y: str, backend: Backend = "cpp") -> int:
    """Calculate the Copy & Paste Edit Distance (CPED)."""

    backend_lower = backend.lower()
    if backend_lower in ("cpp", "c++") and CPP_AVAILABLE:
        return _calculate_cped_cpp(X, Y)
    return _calculate_cped_py(X, Y)


def cped_matrix(
    sequences: List[str], backend: Backend = "cpp", parallel: bool = True
) -> np.ndarray:
    """Calculate the pairwise CPED matrix."""

    backend_lower = backend.lower()
    if backend_lower in ("cpp", "c++") and CPP_AVAILABLE:
        return _calculate_cped_distance_matrix_cpp(sequences, parallel=parallel)
    return _calculate_cped_distance_matrix_py(sequences)


def levd(s1: str, s2: str, backend: Backend = "cpp") -> int:
    """Calculate the Levenshtein Distance."""
    use_cpp = backend in ("cpp", "c++") and CPP_AVAILABLE
    if use_cpp:
        return _calculate_levd_cpp(s1, s2)
    return _calculate_levd_py(s1, s2)


def levd_matrix(
    sequences: List[str], backend: Backend = "cpp", parallel: bool = True
) -> np.ndarray:
    """Calculate the pairwise Levenshtein distance matrix."""
    use_cpp = backend in ("cpp", "c++") and CPP_AVAILABLE
    if use_cpp:
        return _calculate_levd_distance_matrix_cpp(sequences, parallel=parallel)
    return _calculate_levd_distance_matrix_py(sequences)


@overload
def edit_distance(
    a: str,
    b: str,
    distance_type: DistanceType = "levd",
    backend: Backend = "cpp",
    parallel: bool = True,
) -> int: ...


@overload
def edit_distance(
    a: List[str],
    b: None = None,
    distance_type: DistanceType = "levd",
    backend: Backend = "cpp",
    parallel: bool = True,
) -> np.ndarray: ...


def edit_distance(
    a: Union[str, List[str]],
    b: Union[str, None] = None,
    distance_type: DistanceType = "levd",
    backend: Backend = "cpp",
    parallel: bool = True,
) -> Union[int, np.ndarray]:
    """
    Calculates the edit distance between two strings or a matrix of distances
    for a list of strings.

    Args:
        a: The first string or a list of strings.
        b: The second string. If 'a' is a list, this should be None.
        distance_type: The type of distance to calculate ('cped', 'cped-bidir',
            or 'levd'). The 'cped-bidir' option uses the bidirectional CPED
            approximation and always falls back to the Python backend.
        backend: The backend to use for calculation ('cpp', 'c++', or
            'python').
        parallel: Whether to use parallel computation for the distance matrix.

    Returns:
        The edit distance as an integer, or a numpy array distance matrix.
    """
    if isinstance(a, list):
        if b is not None:
            raise ValueError("When 'a' is a list, 'b' must be None.")
        normalized_type = _normalize_distance_type(distance_type)
        if normalized_type == "cped":
            return cped_matrix(a, backend=backend, parallel=parallel)
        if normalized_type == "cped-bidir":
            return _calculate_cped_distance_matrix_py_bidirectional(a)
        if normalized_type == "levd":
            return levd_matrix(a, backend=backend, parallel=parallel)
        raise ValueError(f"Unknown distance_type: {distance_type}")
    elif isinstance(a, str) and isinstance(b, str):
        normalized_type = _normalize_distance_type(distance_type)
        if normalized_type == "cped":
            return cped(a, b, backend=backend)
        if normalized_type == "cped-bidir":
            return _calculate_cped_py_bidirectional(a, b)
        if normalized_type == "levd":
            return levd(a, b, backend=backend)
        raise ValueError(f"Unknown distance_type: {distance_type}")
    else:
        raise TypeError(
            "Inputs 'a' and 'b' must be both strings or 'a' must be a list and 'b' None."
        )


__all__ = [
    "Backend",
    "DistanceType",
    "cped",
    "cped_matrix",
    "levd",
    "levd_matrix",
    "edit_distance",
]
