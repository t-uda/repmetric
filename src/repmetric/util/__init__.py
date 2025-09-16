"""Utility helpers for CPED."""

from __future__ import annotations

import random
import string
from typing import Tuple


def generate_test_case(unit_length: int, repeats: int, edits: int) -> Tuple[str, str]:
    """Generate a pair of strings for scalability testing."""

    chars = string.ascii_lowercase

    def get_random_unit(length: int) -> str:
        return "".join(random.choice(chars) for _ in range(length))

    base_unit = get_random_unit(unit_length)
    string_X = base_unit * repeats
    edited_units = [base_unit] * repeats
    for _ in range(edits):
        edit_type = random.randint(0, 2)
        can_delete = len(edited_units) > 1
        if edit_type == 0 and can_delete:
            idx = random.randrange(len(edited_units))
            edited_units.pop(idx)
        elif edit_type == 1:
            idx = random.randrange(len(edited_units) + 1)
            edited_units.insert(idx, get_random_unit(unit_length))
        else:
            if edited_units:
                idx = random.randrange(len(edited_units))
                edited_units[idx] = get_random_unit(unit_length)
            else:
                edited_units.insert(0, get_random_unit(unit_length))
    string_Y = "".join(edited_units)
    return string_X, string_Y


__all__ = ["generate_test_case"]
