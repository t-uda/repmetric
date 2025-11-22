"""Backend implementations for the repmetric algorithms.

This module loads the optional C++ extension and provides both Python and
C++ implementations for calculating the Copy & Paste Edit Distance (CPED),
Levenshtein distance, and their pairwise distance matrices.
"""

from __future__ import annotations

import ctypes
import glob
import os
from typing import List, Optional, Tuple

import numpy as np

from .geodesic import GeodesicPath


def _load_cpp_lib() -> Tuple[Optional[ctypes.CDLL], bool]:
    """Find and load the compiled C++ shared library."""
    lib_paths = glob.glob(os.path.join(os.path.dirname(__file__), "_cpp.*.so"))
    if not lib_paths:
        lib_paths = glob.glob(os.path.join(os.path.dirname(__file__), "_cpp.so"))
    
    if not lib_paths:
        return None, False

    try:
        repmetric_lib = ctypes.CDLL(lib_paths[0])

        # CPED functions
        repmetric_lib.calculate_cped_cpp_int.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        repmetric_lib.calculate_cped_cpp_int.restype = ctypes.c_int
        repmetric_lib.calculate_cped_distance_matrix_cpp_int.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
            ctypes.c_bool,
        ]
        repmetric_lib.calculate_cped_distance_matrix_cpp_int.restype = None
        
        # CPED Geodesic
        repmetric_lib.calculate_cped_geodesic_cpp.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        repmetric_lib.calculate_cped_geodesic_cpp.restype = ctypes.c_int

        # BICPED functions
        repmetric_lib.calculate_bicped_cpp_int.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        repmetric_lib.calculate_bicped_cpp_int.restype = ctypes.c_int
        repmetric_lib.calculate_bicped_distance_matrix_cpp_int.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
            ctypes.c_bool,
        ]
        repmetric_lib.calculate_bicped_distance_matrix_cpp_int.restype = None

        # Levenshtein functions
        repmetric_lib.calculate_levd_cpp_int.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        repmetric_lib.calculate_levd_cpp_int.restype = ctypes.c_int
        repmetric_lib.calculate_levd_distance_matrix_cpp_int.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
            ctypes.c_bool,
        ]
        repmetric_lib.calculate_levd_distance_matrix_cpp_int.restype = None
        
        # Levenshtein Geodesic
        repmetric_lib.calculate_levd_geodesic_cpp.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        repmetric_lib.calculate_levd_geodesic_cpp.restype = ctypes.c_int

        return repmetric_lib, True
    except (OSError, AttributeError):
        return None, False


repmetric_lib, CPP_AVAILABLE = _load_cpp_lib()


# --- CPED Implementations ---


def _calculate_cped_py(X: str, Y: str) -> int:
    """Calculate CPED using pure Python."""
    dp = _compute_cped_dp_table(X, Y)
    distance = int(dp[len(X)][len(Y)]) if dp[len(X)][len(Y)] != float("inf") else -1
    return distance


def _compute_cped_dp_table(X: str, Y: str, max_copy_len: int = 20) -> List[List[float]]:
    """Return the DP table for the forward CPED approximation."""

    n = len(X)
    m = len(Y)
    dp: List[List[float]] = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    col_mins: List[float] = [float("inf")] * (m + 1)
    col_mins[0] = 1.0

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                col_mins[j] = min(col_mins[j], dp[i][j] + 1.0)
                continue
            current_min = float("inf")
            if i > 0 and j > 0:
                cost = 0.0 if X[i - 1] == Y[j - 1] else 1.0
                current_min = min(current_min, dp[i - 1][j - 1] + cost)
            if j > 0:
                current_min = min(current_min, dp[i][j - 1] + 1.0)
            if i > 0:
                current_min = min(current_min, col_mins[j])
            if j > 1:
                max_len = min(j // 2, max_copy_len)
                for length in range(max_len, 0, -1):
                    substring = Y[j - length : j]
                    if substring and substring in Y[0 : j - length]:
                        cost = dp[i][j - length] + 1.0
                        if cost < current_min:
                            current_min = cost
            dp[i][j] = current_min
            col_mins[j] = min(col_mins[j], dp[i][j] + 1.0)

    return dp


def _calculate_bicped_py(X: str, Y: str) -> int:
    """Calculate CPED using the bidirectional (BICPed) DP refinement."""

    max_copy_len = 20
    forward_dp = _compute_cped_dp_table(X, Y, max_copy_len=max_copy_len)
    reverse_dp = _compute_cped_dp_table(X[::-1], Y[::-1], max_copy_len=max_copy_len)

    n = len(X)
    m = len(Y)

    backward_dp: List[List[float]] = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(m + 1):
            backward_dp[i][j] = reverse_dp[n - i][m - j]

    best = forward_dp[n][m]

    future_repeats: List[List[int]] = [[] for _ in range(m + 1)]
    for j in range(m):
        limit = min(m - j, max_copy_len)
        for length in range(1, limit + 1):
            substring = Y[j : j + length]
            if substring and Y.find(substring, j + 1) != -1:
                future_repeats[j].append(length)

    for i in range(n + 1):
        for j in range(m + 1):
            if forward_dp[i][j] == float("inf"):
                continue
            for length in future_repeats[j]:
                if j + length > m:
                    continue
                if backward_dp[i][j + length] <= float(length):
                    continue
                candidate = forward_dp[i][j] + 1.0 + backward_dp[i][j + length]
                if candidate < best:
                    best = candidate

    return int(best) if best != float("inf") else -1


def _calculate_bicped_cpp(X: str, Y: str) -> int:
    """Wrapper for the C++ BICPed calculation."""
    if not repmetric_lib:
        raise RuntimeError("C++ library not available.")
    X_bytes = X.encode("utf-8")
    Y_bytes = Y.encode("utf-8")
    return repmetric_lib.calculate_bicped_cpp_int(X_bytes, Y_bytes)


def _calculate_cped_cpp(X: str, Y: str) -> int:
    """Wrapper for the C++ CPED calculation."""
    if not repmetric_lib:
        raise RuntimeError("C++ library not available.")
    X_bytes = X.encode("utf-8")
    Y_bytes = Y.encode("utf-8")
    return repmetric_lib.calculate_cped_cpp_int(X_bytes, Y_bytes)


def _calculate_cped_geodesic_cpp(X: str, Y: str) -> Tuple[int, List[str]]:
    """Wrapper for the C++ CPED geodesic calculation."""
    if not repmetric_lib:
        raise RuntimeError("C++ library not available.")
    X_bytes = X.encode("utf-8")
    Y_bytes = Y.encode("utf-8")
    
    # Max path length is bounded, but could be large.
    # n + m is a safe upper bound for simple edits, but with copy/delete it could be less.
    # However, the string representation "C:100" takes more space.
    # Let's allocate a generous buffer.
    max_len = (len(X) + len(Y)) * 10 + 1024
    buffer = ctypes.create_string_buffer(max_len)
    
    dist = repmetric_lib.calculate_cped_geodesic_cpp(X_bytes, Y_bytes, buffer, max_len)
    
    if dist == -1:
        raise RuntimeError("Buffer too small for geodesic path.")
        
    path_str = buffer.value.decode("utf-8")
    if not path_str:
        return dist, GeodesicPath(X, Y, [], dist)
    return dist, GeodesicPath(X, Y, path_str.split(","), dist)


def _calculate_cped_distance_matrix_py(sequences: List[str]) -> np.ndarray:
    """Calculate the pairwise CPED matrix using Python."""
    n = len(sequences)
    dist_matrix = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = _calculate_cped_py(sequences[i], sequences[j])
    return dist_matrix


def _calculate_bicped_distance_matrix_py(
    sequences: List[str],
) -> np.ndarray:
    """Calculate the pairwise BICPed matrix using the bidirectional approximation."""

    n = len(sequences)
    dist_matrix = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = _calculate_bicped_py(sequences[i], sequences[j])
    return dist_matrix


def _calculate_bicped_distance_matrix_cpp(
    sequences: List[str], parallel: bool
) -> np.ndarray:
    """Calculate the pairwise BICPed matrix using C++."""
    if not repmetric_lib:
        raise RuntimeError("C++ library not available.")
    n = len(sequences)
    seq_array = (ctypes.c_char_p * n)()
    encoded_seqs = [s.encode("utf-8") for s in sequences]
    seq_array[:] = encoded_seqs  # type: ignore
    dist_matrix = np.zeros((n, n), dtype=np.int32)
    repmetric_lib.calculate_bicped_distance_matrix_cpp_int(
        seq_array, n, dist_matrix, parallel
    )
    return dist_matrix


def _calculate_cped_distance_matrix_cpp(
    sequences: List[str], parallel: bool
) -> np.ndarray:
    """Calculate the pairwise CPED matrix using C++."""
    if not repmetric_lib:
        raise RuntimeError("C++ library not available.")
    n = len(sequences)
    seq_array = (ctypes.c_char_p * n)()
    encoded_seqs = [s.encode("utf-8") for s in sequences]
    seq_array[:] = encoded_seqs  # type: ignore
    dist_matrix = np.zeros((n, n), dtype=np.int32)
    repmetric_lib.calculate_cped_distance_matrix_cpp_int(
        seq_array, n, dist_matrix, parallel
    )
    return dist_matrix


# --- Levenshtein Implementations ---


def _calculate_levd_py(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance using pure Python."""
    n, m = len(s1), len(s2)
    if n > m:
        s1, s2 = s2, s1
        n, m = m, n

    current_row = list(range(n + 1))
    for i in range(1, m + 1):
        previous_row = current_row
        current_row = [i] + [0] * n
        for j in range(1, n + 1):
            add, delete, change = (
                previous_row[j] + 1,
                current_row[j - 1] + 1,
                previous_row[j - 1],
            )
            if s1[j - 1] != s2[i - 1]:
                change += 1
            current_row[j] = min(add, delete, change)
    return current_row[n]


def _calculate_levd_geodesic_py(s1: str, s2: str) -> Tuple[int, GeodesicPath]:
    """Calculate Levenshtein geodesic using pure Python."""
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
            
    path = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                path.append("M" if cost == 0 else "S")
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            path.append("D")
            i -= 1
            continue
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            path.append("I")
            j -= 1
            continue
            
    return dp[n][m], GeodesicPath(s1, s2, path[::-1], dp[n][m])


def _calculate_levd_cpp(s1: str, s2: str) -> int:
    """Wrapper for the C++ Levenshtein calculation."""
    if not repmetric_lib:
        raise RuntimeError("C++ library not available.")
    s1_bytes = s1.encode("utf-8")
    s2_bytes = s2.encode("utf-8")
    return repmetric_lib.calculate_levd_cpp_int(s1_bytes, s2_bytes)


def _calculate_levd_geodesic_cpp(s1: str, s2: str) -> Tuple[int, GeodesicPath]:
    """Wrapper for the C++ Levenshtein geodesic calculation."""
    if not repmetric_lib:
        raise RuntimeError("C++ library not available.")
    s1_bytes = s1.encode("utf-8")
    s2_bytes = s2.encode("utf-8")
    
    # Max path length is n + m
    max_len = len(s1) + len(s2) + 1
    buffer = ctypes.create_string_buffer(max_len)
    
    dist = repmetric_lib.calculate_levd_geodesic_cpp(s1_bytes, s2_bytes, buffer, max_len)
    
    if dist == -1:
        raise RuntimeError("Buffer too small for geodesic path.")
        
    path_str = buffer.value.decode("utf-8")
    # Parse path string "M,S,I,D" -> ["M", "S", "I", "D"]
    # Actually the C++ implementation returns "MSID" without commas?
    # Let's check levd.cpp again.
    # path += (cost == 0 ? 'M' : 'S');
    # path += 'D';
    # path += 'I';
    # So it returns a string like "MMID".
    
    return dist, GeodesicPath(s1, s2, list(path_str), dist)


def _calculate_levd_distance_matrix_py(sequences: List[str]) -> np.ndarray:
    """Calculate the pairwise Levenshtein matrix using Python."""
    n = len(sequences)
    dist_matrix = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i, n):
            dist = _calculate_levd_py(sequences[i], sequences[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix


def _calculate_levd_distance_matrix_cpp(
    sequences: List[str], parallel: bool
) -> np.ndarray:
    """Calculate the pairwise Levenshtein matrix using C++."""
    if not repmetric_lib:
        raise RuntimeError("C++ library not available.")
    n = len(sequences)
    seq_array = (ctypes.c_char_p * n)()
    encoded_seqs = [s.encode("utf-8") for s in sequences]
    seq_array[:] = encoded_seqs  # type: ignore
    dist_matrix = np.zeros((n, n), dtype=np.int32)
    repmetric_lib.calculate_levd_distance_matrix_cpp_int(
        seq_array, n, dist_matrix, parallel
    )
    return dist_matrix
