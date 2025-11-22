import pytest
from repmetric.api import levd, cped, edit_distance
from repmetric.geodesic import GeodesicPath


def test_levd_geodesic_simple():
    s1 = "kitten"
    s2 = "sitting"
    # k -> s (S)
    # i -> i (M)
    # t -> t (M)
    # t -> t (M)
    # e -> i (S)
    # n -> n (M)
    #   -> g (I)
    # Distance 3.
    # Path could be SMMMSMI or similar.

    dist, path_obj = levd(s1, s2, geodesic=True, backend="python")
    assert dist == 3
    assert isinstance(path_obj, GeodesicPath)
    assert len(path_obj.get_operations()) == 7

    path_strings = path_obj.get_path_strings()
    assert path_strings[0] == s1
    assert path_strings[-1] == s2
    # Verify intermediate steps
    # "kitten" -> "sitten" (S) -> "sitten" (M) -> ...
    # My implementation of get_path_strings might include duplicates if "M" is treated as no-op.
    # Let's check the implementation.
    # M: processed.append(start[i]), i++, j++. No string change.
    # S: processed.append(end[j]), i++, j++. String changed.
    # So M does not append to path_strings.
    # S, I, D append to path_strings.
    # So path_strings should show the evolution.

    # Check length of path_strings
    # 3 edits -> 3 changes -> 4 strings (start + 3 intermediate)
    assert len(path_strings) == 4
    assert path_strings == ["kitten", "sitten", "sittin", "sitting"]


def test_levd_geodesic_cpp():
    s1 = "kitten"
    s2 = "sitting"
    try:
        dist, path_obj = levd(s1, s2, geodesic=True, backend="cpp")
    except RuntimeError:
        pytest.skip("C++ backend not available")

    assert dist == 3
    assert isinstance(path_obj, GeodesicPath)

    path_strings = path_obj.get_path_strings()
    assert path_strings[0] == s1
    assert path_strings[-1] == s2
    assert len(path_strings) == 4


def verify_path(s1, s2, path):
    i, j = 0, 0
    for op in path:
        if op == "M":
            if i >= len(s1) or j >= len(s2) or s1[i] != s2[j]:
                return False
            i += 1
            j += 1
        elif op == "S":
            if i >= len(s1) or j >= len(s2) or s1[i] == s2[j]:
                return False
            i += 1
            j += 1
        elif op == "I":
            if j >= len(s2):
                return False
            j += 1
        elif op == "D":
            if i >= len(s1):
                return False
            i += 1
        else:
            return False
    return i == len(s1) and j == len(s2)


def test_levd_geodesic_empty():
    s1 = ""
    s2 = "abc"
    dist, path_obj = levd(s1, s2, geodesic=True, backend="python")
    assert dist == 3
    assert path_obj.get_operations() == ["I", "I", "I"]
    assert path_obj.get_path_strings() == ["", "a", "ab", "abc"]

    s1 = "abc"
    s2 = ""
    dist, path_obj = levd(s1, s2, geodesic=True, backend="python")
    assert dist == 3
    assert path_obj.get_operations() == ["D", "D", "D"]
    assert path_obj.get_path_strings() == ["abc", "bc", "c", ""]


def test_cped_geodesic_cpp():
    s1 = "kitten"
    s2 = "sitting"
    try:
        dist, path_obj = cped(s1, s2, geodesic=True, backend="cpp")
    except RuntimeError:
        pytest.skip("C++ backend not available")

    # CPED might be same as Levenshtein for this case
    assert dist == 3
    path_strings = path_obj.get_path_strings()
    assert path_strings[0] == s1
    assert path_strings[-1] == s2


def test_cped_geodesic_copy():
    # Test case where Copy is beneficial
    # X = "abcde"
    # Y = "abcdeabcde"
    # Cost: 5 matches + 1 copy (len 5) = 1? No, Copy cost is 1.
    # Matches cost 0.
    # So cost should be 1.
    # Path: M,M,M,M,M, C:5

    s1 = "abcde"
    s2 = "abcdeabcde"
    try:
        dist, path_obj = cped(s1, s2, geodesic=True, backend="cpp")
    except RuntimeError:
        pytest.skip("C++ backend not available")

    assert dist == 1
    ops = path_obj.get_operations()
    # Ops might be M,M,M,M,M, C:5
    assert "C:5" in ops

    path_strings = path_obj.get_path_strings()
    assert path_strings[0] == s1
    assert path_strings[-1] == s2
    # "abcde" -> "abcdeabcde" (Copy)
    # Matches don't change string.
    # Copy adds "abcde".
    assert len(path_strings) == 2
    assert path_strings == ["abcde", "abcdeabcde"]


# Additional comprehensive tests


def test_levd_geodesic_identical_strings():
    """Test geodesic for identical strings."""
    s1 = "hello"
    s2 = "hello"
    dist, path_obj = levd(s1, s2, geodesic=True, backend="python")
    assert dist == 0
    # All operations should be matches
    ops = path_obj.get_operations()
    assert all(op == "M" for op in ops)
    assert len(ops) == 5
    # Path strings should only contain the original string
    path_strings = path_obj.get_path_strings()
    assert len(path_strings) == 1
    assert path_strings == ["hello"]


def test_levd_geodesic_single_char():
    """Test geodesic for single character strings."""
    # Substitution
    dist, path_obj = levd("a", "b", geodesic=True, backend="python")
    assert dist == 1
    assert path_obj.get_operations() == ["S"]
    assert path_obj.get_path_strings() == ["a", "b"]

    # Deletion
    dist, path_obj = levd("a", "", geodesic=True, backend="python")
    assert dist == 1
    assert path_obj.get_operations() == ["D"]
    assert path_obj.get_path_strings() == ["a", ""]

    # Insertion
    dist, path_obj = levd("", "a", geodesic=True, backend="python")
    assert dist == 1
    assert path_obj.get_operations() == ["I"]
    assert path_obj.get_path_strings() == ["", "a"]


def test_levd_geodesic_only_insertions():
    """Test geodesic with only insertions."""
    s1 = "abc"
    s2 = "abcdef"
    dist, path_obj = levd(s1, s2, geodesic=True, backend="python")
    assert dist == 3
    ops = path_obj.get_operations()
    assert ops.count("M") == 3
    assert ops.count("I") == 3
    path_strings = path_obj.get_path_strings()
    assert path_strings[0] == s1
    assert path_strings[-1] == s2


def test_levd_geodesic_only_deletions():
    """Test geodesic with only deletions."""
    s1 = "abcdef"
    s2 = "abc"
    dist, path_obj = levd(s1, s2, geodesic=True, backend="python")
    assert dist == 3
    ops = path_obj.get_operations()
    assert ops.count("M") == 3
    assert ops.count("D") == 3
    path_strings = path_obj.get_path_strings()
    assert path_strings[0] == s1
    assert path_strings[-1] == s2


def test_levd_geodesic_only_substitutions():
    """Test geodesic with only substitutions."""
    s1 = "abc"
    s2 = "xyz"
    dist, path_obj = levd(s1, s2, geodesic=True, backend="python")
    assert dist == 3
    ops = path_obj.get_operations()
    assert ops == ["S", "S", "S"]
    path_strings = path_obj.get_path_strings()
    assert path_strings == ["abc", "xbc", "xyc", "xyz"]


def test_cped_geodesic_block_delete():
    """Test CPED geodesic with block delete operation."""
    s1 = "abcdefgh"
    s2 = "ah"
    try:
        dist, path_obj = cped(s1, s2, geodesic=True, backend="cpp")
    except RuntimeError:
        pytest.skip("C++ backend not available")

    # Should use block delete
    ops = path_obj.get_operations()
    # Check if block delete is used
    has_block_delete = any(op.startswith("D:") for op in ops)
    assert has_block_delete or dist <= 2  # Either uses block delete or is efficient
    path_strings = path_obj.get_path_strings()
    assert path_strings[0] == s1
    assert path_strings[-1] == s2


def test_cped_geodesic_multiple_copies():
    """Test CPED geodesic with multiple copy operations."""
    s1 = "ab"
    s2 = "ababab"
    try:
        dist, path_obj = cped(s1, s2, geodesic=True, backend="cpp")
    except RuntimeError:
        pytest.skip("C++ backend not available")

    # Should use copy operations
    ops = path_obj.get_operations()
    copy_ops = [op for op in ops if op.startswith("C:")]
    assert len(copy_ops) >= 1  # At least one copy operation
    assert dist <= 2  # Should be efficient
    path_strings = path_obj.get_path_strings()
    assert path_strings[0] == s1
    assert path_strings[-1] == s2


def test_geodesic_path_attributes():
    """Test GeodesicPath object attributes."""
    s1 = "hello"
    s2 = "world"
    dist, path_obj = levd(s1, s2, geodesic=True, backend="python")

    assert path_obj.start_string == s1
    assert path_obj.end_string == s2
    assert path_obj.distance == dist
    assert isinstance(path_obj.get_operations(), list)
    assert isinstance(path_obj.get_path_strings(), list)


def test_geodesic_via_edit_distance():
    """Test geodesic through edit_distance function."""
    s1 = "test"
    s2 = "text"

    # Levenshtein
    dist, path_obj = edit_distance(s1, s2, distance_type="levd", geodesic=True)
    assert dist == 1
    assert isinstance(path_obj, GeodesicPath)
    assert path_obj.get_path_strings()[0] == s1
    assert path_obj.get_path_strings()[-1] == s2

    # CPED
    try:
        dist, path_obj = edit_distance(s1, s2, distance_type="cped", geodesic=True)
        assert dist == 1
        assert isinstance(path_obj, GeodesicPath)
    except (RuntimeError, NotImplementedError):
        pytest.skip("C++ backend not available for CPED geodesic")


def test_geodesic_bicped_not_implemented():
    """Test that geodesic raises error for BICPed."""
    with pytest.raises(NotImplementedError):
        edit_distance("test", "text", distance_type="bicped", geodesic=True)


def test_levd_geodesic_consistency_python_cpp():
    """Test that Python and C++ backends produce consistent results."""
    test_cases = [
        ("hello", "world"),
        ("kitten", "sitting"),
        ("", "abc"),
        ("abc", ""),
        ("same", "same"),
    ]

    for s1, s2 in test_cases:
        dist_py, path_py = levd(s1, s2, geodesic=True, backend="python")

        try:
            dist_cpp, path_cpp = levd(s1, s2, geodesic=True, backend="cpp")
        except RuntimeError:
            pytest.skip("C++ backend not available")

        # Distances should match
        assert dist_py == dist_cpp, f"Distance mismatch for ({s1}, {s2})"

        # Path strings should match
        assert (
            path_py.get_path_strings() == path_cpp.get_path_strings()
        ), f"Path mismatch for ({s1}, {s2})"


def test_geodesic_path_reconstruction_correctness():
    """Verify that path reconstruction produces valid intermediate strings."""
    test_cases = [
        ("abc", "def"),
        ("hello", "hallo"),
        ("test", "testing"),
        ("remove", "rem"),
    ]

    for s1, s2 in test_cases:
        dist, path_obj = levd(s1, s2, geodesic=True, backend="python")
        path_strings = path_obj.get_path_strings()

        # First string should be start
        assert path_strings[0] == s1

        # Last string should be end
        assert path_strings[-1] == s2

        # Each consecutive pair should differ by at most 1 edit
        for i in range(len(path_strings) - 1):
            intermediate_dist = levd(
                path_strings[i], path_strings[i + 1], geodesic=False
            )
            assert (
                intermediate_dist <= 1
            ), f"Invalid step from {path_strings[i]} to {path_strings[i + 1]}"
