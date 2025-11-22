import pytest
from repmetric.api import levd, cped
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
        if op == 'M':
            if i >= len(s1) or j >= len(s2) or s1[i] != s2[j]:
                return False
            i += 1
            j += 1
        elif op == 'S':
            if i >= len(s1) or j >= len(s2) or s1[i] == s2[j]:
                return False
            i += 1
            j += 1
        elif op == 'I':
            if j >= len(s2):
                return False
            j += 1
        elif op == 'D':
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
    assert path_obj.get_operations() == ['I', 'I', 'I']
    assert path_obj.get_path_strings() == ["", "a", "ab", "abc"]
    
    s1 = "abc"
    s2 = ""
    dist, path_obj = levd(s1, s2, geodesic=True, backend="python")
    assert dist == 3
    assert path_obj.get_operations() == ['D', 'D', 'D']
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
