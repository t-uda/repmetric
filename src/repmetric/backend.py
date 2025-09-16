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


def _load_cpp_lib() -> Tuple[Optional[ctypes.CDLL], bool]:
    """Find and load the compiled C++ shared library."""
    lib_paths = glob.glob(os.path.join(os.path.dirname(__file__), "_cpp.*.so"))
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
        ]
        repmetric_lib.calculate_cped_distance_matrix_cpp_int.restype = None

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
        ]
        repmetric_lib.calculate_levd_distance_matrix_cpp_int.restype = None

        return repmetric_lib, True
    except (OSError, AttributeError):
        return None, False


repmetric_lib, CPP_AVAILABLE = _load_cpp_lib()


# --- CPED Implementations ---


def _calculate_cped_py(X: str, Y: str) -> int:
    """Calculate CPED using pure Python."""
    n = len(X)
    m = len(Y)
    dp: List[List[float]] = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    col_mins: List[float] = [float("inf")] * (m + 1)
    col_mins[0] = 1.0

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            current_min = float("inf")
            if i > 0 and j > 0:
                cost = 0.0 if X[i - 1] == Y[j - 1] else 1.0
                current_min = min(current_min, dp[i - 1][j - 1] + cost)
            if j > 0:
                current_min = min(current_min, dp[i][j - 1] + 1)
            if i > 0:
                current_min = min(current_min, col_mins[j])
            if j > 1:
                max_len = min(j // 2, 20)
                for length in range(max_len, 0, -1):
                    substring = Y[j - length : j]
                    search_space = Y[0 : j - length]
                    if substring in search_space:
                        cost = dp[i][j - length] + 1.0
                        if cost < current_min:
                            current_min = cost
            dp[i][j] = current_min
            col_mins[j] = min(col_mins[j], dp[i][j] + 1.0)

    distance = int(dp[n][m]) if dp[n][m] != float("inf") else -1
    return distance


def _calculate_cped_cpp(X: str, Y: str) -> int:
    """Wrapper for the C++ CPED calculation."""
    if not repmetric_lib:
        raise RuntimeError("C++ library not available.")
    X_bytes = X.encode("utf-8")
    Y_bytes = Y.encode("utf-8")
    return repmetric_lib.calculate_cped_cpp_int(X_bytes, Y_bytes)


def _calculate_cped_distance_matrix_py(sequences: List[str]) -> np.ndarray:
    """Calculate the pairwise CPED matrix using Python."""
    n = len(sequences)
    dist_matrix = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = _calculate_cped_py(sequences[i], sequences[j])
    return dist_matrix


def _calculate_cped_distance_matrix_cpp(sequences: List[str]) -> np.ndarray:
    """Calculate the pairwise CPED matrix using C++."""
    if not repmetric_lib:
        raise RuntimeError("C++ library not available.")
    n = len(sequences)
    seq_array = (ctypes.c_char_p * n)()
    encoded_seqs = [s.encode("utf-8") for s in sequences]
    seq_array[:] = encoded_seqs  # type: ignore
    dist_matrix = np.zeros((n, n), dtype=np.int32)
    repmetric_lib.calculate_cped_distance_matrix_cpp_int(seq_array, n, dist_matrix)
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


def _calculate_levd_cpp(s1: str, s2: str) -> int:
    """Wrapper for the C++ Levenshtein calculation."""
    if not repmetric_lib:
        raise RuntimeError("C++ library not available.")
    s1_bytes = s1.encode("utf-8")
    s2_bytes = s2.encode("utf-8")
    return repmetric_lib.calculate_levd_cpp_int(s1_bytes, s2_bytes)


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


def _calculate_levd_distance_matrix_cpp(sequences: List[str]) -> np.ndarray:
    """Calculate the pairwise Levenshtein matrix using C++."""
    if not repmetric_lib:
        raise RuntimeError("C++ library not available.")
    n = len(sequences)
    seq_array = (ctypes.c_char_p * n)()
    encoded_seqs = [s.encode("utf-8") for s in sequences]
    seq_array[:] = encoded_seqs  # type: ignore
    dist_matrix = np.zeros((n, n), dtype=np.int32)
    repmetric_lib.calculate_levd_distance_matrix_cpp_int(seq_array, n, dist_matrix)
    return dist_matrix
