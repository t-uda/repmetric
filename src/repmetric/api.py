"""High level API for repmetric operations."""

from __future__ import annotations

from typing import List, Literal, Union, overload

import numpy as np

from .backend import (
    CPP_AVAILABLE,
    _calculate_bicped_distance_matrix_py,
    _calculate_bicped_distance_matrix_cpp,
    _calculate_bicped_py,
    _calculate_bicped_cpp,
    _calculate_cped_cpp,
    _calculate_cped_distance_matrix_cpp,
    _calculate_cped_distance_matrix_py,
    _calculate_cped_py,
    _calculate_levd_cpp,
    _calculate_levd_distance_matrix_cpp,
    _calculate_levd_distance_matrix_py,
    _calculate_levd_distance_matrix_py,
    _calculate_levd_py,
    _calculate_levd_geodesic_cpp,
    _calculate_cped_geodesic_cpp,
)
from .geodesic import GeodesicPath

Backend = Literal["cpp", "c++", "python"]
DistanceType = Literal[
    "cped",
    "levd",
    "bicped",
]


def cped(
    X: str, Y: str, backend: Backend = "cpp", geodesic: bool = False
) -> Union[int, Tuple[int, GeodesicPath]]:
    """Calculate the Copy & Paste Edit Distance (CPED)."""

    backend_lower = backend.lower()
    if backend_lower in ("cpp", "c++"):
        if CPP_AVAILABLE:
            if geodesic:
                return _calculate_cped_geodesic_cpp(X, Y)
            return _calculate_cped_cpp(X, Y)
        if geodesic:
            raise NotImplementedError("Geodesic calculation for CPED requires C++ backend.")
        return _calculate_cped_py(X, Y)
    if backend_lower == "python":
        return _calculate_cped_py(X, Y)
    raise ValueError(f"Unknown backend: {backend}")


def bicped(X: str, Y: str, backend: Backend = "cpp") -> int:
    """Calculate the bidirectional CPED approximation (BICPed)."""

    backend_lower = backend.lower()
    if backend_lower in ("cpp", "c++"):
        if CPP_AVAILABLE:
            return _calculate_bicped_cpp(X, Y)
        return _calculate_bicped_py(X, Y)
    if backend_lower == "python":
        return _calculate_bicped_py(X, Y)
    raise ValueError(f"Unknown backend: {backend}")


def cped_matrix(
    sequences: List[str], backend: Backend = "cpp", parallel: bool = True
) -> np.ndarray:
    """Calculate the pairwise CPED matrix."""

    backend_lower = backend.lower()
    if backend_lower in ("cpp", "c++"):
        if CPP_AVAILABLE:
            return _calculate_cped_distance_matrix_cpp(sequences, parallel=parallel)
        return _calculate_cped_distance_matrix_py(sequences)
    if backend_lower == "python":
        return _calculate_cped_distance_matrix_py(sequences)
    raise ValueError(f"Unknown backend: {backend}")


def bicped_matrix(
    sequences: List[str], backend: Backend = "cpp", parallel: bool = True
) -> np.ndarray:
    """Calculate the pairwise bidirectional CPED (BICPed) matrix."""

    backend_lower = backend.lower()
    if backend_lower in ("cpp", "c++"):
        if CPP_AVAILABLE:
            return _calculate_bicped_distance_matrix_cpp(sequences, parallel=parallel)
        return _calculate_bicped_distance_matrix_py(sequences)
    if backend_lower == "python":
        return _calculate_bicped_distance_matrix_py(sequences)
    raise ValueError(f"Unknown backend: {backend}")


def levd(
    s1: str, s2: str, backend: Backend = "cpp", geodesic: bool = False
) -> Union[int, Tuple[int, GeodesicPath]]:
    """Calculate the Levenshtein Distance."""
    use_cpp = backend in ("cpp", "c++") and CPP_AVAILABLE
    if use_cpp:
        if geodesic:
            return _calculate_levd_geodesic_cpp(s1, s2)
        return _calculate_levd_cpp(s1, s2)
    
    if geodesic:
        # Fallback or raise error if python version not implemented
        from .backend import _calculate_levd_geodesic_py
        return _calculate_levd_geodesic_py(s1, s2)
        
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
    geodesic: bool = False,
) -> Union[int, Tuple[int, GeodesicPath]]: ...


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
    geodesic: bool = False,
) -> Union[int, np.ndarray, Tuple[int, GeodesicPath]]:
    """
    Calculates the edit distance between two strings or a matrix of distances
    for a list of strings.

    Args:
        a: The first string or a list of strings.
        b: The second string. If 'a' is a list, this should be None.
        distance_type: The type of distance to calculate.
            Supported values are 'cped', 'levd', and 'bicped' (the
            bidirectional CPED approximation).
        backend: The backend to use for calculation ('cpp', 'python').
        parallel: Whether to use parallel computation for the distance matrix.

    Returns:
        The edit distance as an integer, or a numpy array distance matrix.
    """
    if isinstance(a, list):
        if b is not None:
            raise ValueError("When 'a' is a list, 'b' must be None.")
        distance_type_lower = distance_type.lower()
        if distance_type_lower == "cped":
            return cped_matrix(a, backend=backend, parallel=parallel)
        if distance_type_lower == "bicped":
            return bicped_matrix(a, backend=backend, parallel=parallel)
        if distance_type_lower == "levd":
            return levd_matrix(a, backend=backend, parallel=parallel)
        raise ValueError(f"Unknown distance_type: {distance_type}")
    elif isinstance(a, str) and isinstance(b, str):
        distance_type_lower = distance_type.lower()
        if distance_type_lower == "cped":
            return cped(a, b, backend=backend, geodesic=geodesic)
        if distance_type_lower == "bicped":
            if geodesic:
                raise NotImplementedError("Geodesic not implemented for BICPed")
            return bicped(a, b, backend=backend)
        if distance_type_lower == "levd":
            return levd(a, b, backend=backend, geodesic=geodesic)
        raise ValueError(f"Unknown distance_type: {distance_type}")
    else:
        raise TypeError(
            "Inputs 'a' and 'b' must be both strings or 'a' must be a list and 'b' None."
        )


__all__ = [
    "Backend",
    "DistanceType",
    "cped",
    "bicped",
    "cped_matrix",
    "bicped_matrix",
    "levd",
    "levd_matrix",
    "edit_distance",
]
