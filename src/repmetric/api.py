"""High level API for repmetric operations."""

from __future__ import annotations

from typing import List, Literal, Union, overload

import numpy as np

from .backend import (
    CPP_AVAILABLE,
    _calculate_cped_cpp,
    _calculate_cped_distance_matrix_cpp,
    _calculate_cped_distance_matrix_py,
    _calculate_cped_py,
    _calculate_levd_cpp,
    _calculate_levd_distance_matrix_cpp,
    _calculate_levd_distance_matrix_py,
    _calculate_levd_py,
)

Backend = Literal["cpp", "c++", "python"]
DistanceType = Literal["cped", "levd"]


def cped(X: str, Y: str, backend: Backend = "cpp") -> int:
    """Calculate the Copy & Paste Edit Distance (CPED)."""
    use_cpp = backend in ("cpp", "c++") and CPP_AVAILABLE
    if use_cpp:
        return _calculate_cped_cpp(X, Y)
    return _calculate_cped_py(X, Y)


def cped_matrix(sequences: List[str], backend: Backend = "cpp") -> np.ndarray:
    """Calculate the pairwise CPED matrix."""
    use_cpp = backend in ("cpp", "c++") and CPP_AVAILABLE
    if use_cpp:
        return _calculate_cped_distance_matrix_cpp(sequences)
    return _calculate_cped_distance_matrix_py(sequences)


def levd(s1: str, s2: str, backend: Backend = "cpp") -> int:
    """Calculate the Levenshtein Distance."""
    use_cpp = backend in ("cpp", "c++") and CPP_AVAILABLE
    if use_cpp:
        return _calculate_levd_cpp(s1, s2)
    return _calculate_levd_py(s1, s2)


def levd_matrix(sequences: List[str], backend: Backend = "cpp") -> np.ndarray:
    """Calculate the pairwise Levenshtein distance matrix."""
    use_cpp = backend in ("cpp", "c++") and CPP_AVAILABLE
    if use_cpp:
        return _calculate_levd_distance_matrix_cpp(sequences)
    return _calculate_levd_distance_matrix_py(sequences)


@overload
def edit_distance(
    a: str,
    b: str,
    distance_type: DistanceType = "levd",
    backend: Backend = "cpp",
) -> int: ...


@overload
def edit_distance(
    a: List[str],
    b: None = None,
    distance_type: DistanceType = "levd",
    backend: Backend = "cpp",
) -> np.ndarray: ...


def edit_distance(
    a: Union[str, List[str]],
    b: Union[str, None] = None,
    distance_type: DistanceType = "levd",
    backend: Backend = "cpp",
) -> Union[int, np.ndarray]:
    """
    Calculates the edit distance between two strings or a matrix of distances
    for a list of strings.

    Args:
        a: The first string or a list of strings.
        b: The second string. If 'a' is a list, this should be None.
        distance_type: The type of distance to calculate ('cped' or 'levd').
        backend: The backend to use for calculation ('cpp' or 'python').

    Returns:
        The edit distance as an integer, or a numpy array distance matrix.
    """
    if isinstance(a, list):
        if b is not None:
            raise ValueError("When 'a' is a list, 'b' must be None.")
        if distance_type == "cped":
            return cped_matrix(a, backend=backend)
        elif distance_type == "levd":
            return levd_matrix(a, backend=backend)
        else:
            raise ValueError(f"Unknown distance_type: {distance_type}")
    elif isinstance(a, str) and isinstance(b, str):
        if distance_type == "cped":
            return cped(a, b, backend=backend)
        elif distance_type == "levd":
            return levd(a, b, backend=backend)
        else:
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
