"""High level API for repmetric operations."""

from __future__ import annotations

from typing import List, Literal, Tuple, Union, overload

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
    _calculate_cped_geodesic_py,
    _calculate_cped_py,
    _calculate_levd_cpp,
    _calculate_levd_distance_matrix_cpp,
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
    "levenshtein",
    "bicped",
]


def cped(
    X: str, Y: str, backend: Backend = "cpp", geodesic: bool = False
) -> Union[int, Tuple[int, GeodesicPath]]:
    """Compute the Copy & Paste Edit Distance (CPED) between two strings.

    Args:
        X: The source string.
        Y: The target string.
        backend: Execution backend. ``"cpp"`` (default) attempts to use the
            compiled implementation and falls back to the Python
            implementation if the extension is unavailable. ``"python"`` forces
            the pure Python implementation.
        geodesic: If True, returns a tuple of (distance, GeodesicPath) containing
            the edit distance and the geodesic path. If False, returns only the
            distance.

    Returns:
        The CPED between ``X`` and ``Y`` if geodesic is False, or a tuple of
        (distance, GeodesicPath) if geodesic is True.

    Raises:
        ValueError: If an unknown backend name is provided.
    """

    backend_lower = backend.lower()
    if backend_lower in ("cpp", "c++"):
        if CPP_AVAILABLE:
            if geodesic:
                return _calculate_cped_geodesic_cpp(X, Y)
            return _calculate_cped_cpp(X, Y)
        if geodesic:
            return _calculate_cped_geodesic_py(X, Y)
        return _calculate_cped_py(X, Y)
    if backend_lower == "python":
        if geodesic:
            return _calculate_cped_geodesic_py(X, Y)
        return _calculate_cped_py(X, Y)
    raise ValueError(f"Unknown backend: {backend}")


def bicped(X: str, Y: str, backend: Backend = "cpp") -> int:
    """Compute the bidirectional CPED approximation (BICPed).

    Args:
        X: The source string.
        Y: The target string.
        backend: Execution backend. ``"cpp"`` (default) attempts to use the
            compiled implementation and falls back to the Python
            implementation if the extension is unavailable. ``"python"`` forces
            the pure Python implementation.

    Returns:
        The BICPed distance between ``X`` and ``Y``.

    Raises:
        ValueError: If an unknown backend name is provided.
    """

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
    """Compute the pairwise CPED matrix for a collection of strings.

    Args:
        sequences: Strings for which to calculate the pairwise distances.
        backend: Execution backend. ``"cpp"`` (default) attempts to use the
            compiled implementation and falls back to the Python
            implementation if the extension is unavailable. ``"python"`` forces
            the pure Python implementation.
        parallel: When ``True`` (default) and the C++ backend is available, the
            distance matrix is computed in parallel.

    Returns:
        A square numpy array containing the CPED distances.

    Raises:
        ValueError: If an unknown backend name is provided.
    """

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
    """Compute the pairwise bidirectional CPED (BICPed) distance matrix.

    Args:
        sequences: Strings for which to calculate the pairwise distances.
        backend: Execution backend. ``"cpp"`` (default) attempts to use the
            compiled implementation and falls back to the Python
            implementation if the extension is unavailable. ``"python"`` forces
            the pure Python implementation.
        parallel: When ``True`` (default) and the C++ backend is available, the
            distance matrix is computed in parallel.

    Returns:
        A square numpy array containing the BICPed distances.

    Raises:
        ValueError: If an unknown backend name is provided.
    """

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
    """Compute the Levenshtein distance between two strings.

    Args:
        s1: The source string.
        s2: The target string.
        backend: Execution backend. ``"cpp"`` (default) attempts to use the
            compiled implementation and falls back to the Python
            implementation if the extension is unavailable. Any other value
            selects the pure Python implementation.
        geodesic: If True, returns a tuple of (distance, GeodesicPath) containing
            the edit distance and the geodesic path. If False, returns only the
            distance.

    Returns:
        The Levenshtein distance between ``s1`` and ``s2`` if geodesic is False,
        or a tuple of (distance, GeodesicPath) if geodesic is True.
    """
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
    """Compute the pairwise Levenshtein distance matrix.

    Args:
        sequences: Strings for which to calculate the pairwise distances.
        backend: Execution backend. ``"cpp"`` (default) attempts to use the
            compiled implementation and falls back to the Python
            implementation if the extension is unavailable. Any other value
            selects the pure Python implementation.
        parallel: When ``True`` (default) and the C++ backend is available, the
            distance matrix is computed in parallel.

    Returns:
        A square numpy array containing the Levenshtein distances.
    """
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
    """Compute an edit distance or a distance matrix for the provided input.

    This helper routes to :func:`cped`, :func:`levd`, :func:`bicped`, and their
    corresponding matrix functions based on the input type and the
    ``distance_type`` argument.

    Args:
        a: Either the first string to compare or a list of strings for a
            distance matrix.
        b: The second string to compare. Must be ``None`` when ``a`` is a list.
        distance_type: Which edit distance variant to compute. Accepted values
            are ``"cped"``, ``"levd"``/``"levenshtein"``, and ``"bicped"``.
        backend: Execution backend. ``"cpp"`` (default) attempts to use the
            compiled implementation and falls back to the Python
            implementation if the extension is unavailable. ``"python"`` forces
            the pure Python implementation.
        parallel: When ``True`` (default) and a C++ backend matrix routine is
            available, compute the matrix in parallel.
        geodesic: If True, returns a tuple of (distance, GeodesicPath) when
            computing distance between two strings. Only supported for ``"cped"``
            and ``"levd"`` distance types. Ignored when computing distance matrices.

    Returns:
        An integer when both ``a`` and ``b`` are strings and geodesic is False,
        a tuple of (distance, GeodesicPath) when both are strings and geodesic is True,
        or a numpy array containing the pairwise distances for ``a`` when ``a`` is a list.

    Raises:
        TypeError: If the combination of ``a`` and ``b`` is invalid.
        ValueError: If ``distance_type`` is unknown or the selected backend
            rejects the provided backend name.
        NotImplementedError: If geodesic is True for ``"bicped"`` distance type.
    """
    if isinstance(a, list):
        if b is not None:
            raise ValueError("When 'a' is a list, 'b' must be None.")
        distance_type_lower = distance_type.lower()
        if distance_type_lower == "cped":
            return cped_matrix(a, backend=backend, parallel=parallel)
        if distance_type_lower == "bicped":
            return bicped_matrix(a, backend=backend, parallel=parallel)
        if distance_type_lower in ("levd", "levenshtein"):
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
        if distance_type_lower in ("levd", "levenshtein"):
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
