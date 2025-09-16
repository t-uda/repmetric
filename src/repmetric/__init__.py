"""A library for efficient string distance and sequence similarity metrics."""

from __future__ import annotations

from . import util
from .api import (
    Backend,
    DistanceType,
    cped,
    cped_matrix,
    edit_distance,
    levd,
    levd_matrix,
)
from .backend import CPP_AVAILABLE

__all__ = [
    "edit_distance",
    "cped",
    "cped_matrix",
    "levd",
    "levd_matrix",
    "Backend",
    "DistanceType",
    "CPP_AVAILABLE",
    "util",
]
