import numpy as np
import pytest
import repmetric
from unittest.mock import patch

# --- Test Cases ---
CPED_TEST_CASES = [
    ("", "", 0),
    ("", "abc", 3),
    ("abc", "", 1),
    ("abcdef", "abcdef", 0),
    ("abc", "axc", 1),
    ("abcdef", "axydyf", 3),
    ("ac", "abc", 1),
    ("abc", "ac", 1),
    ("ab", "abab", 1),
    ("axc", "abcabc", 2),
]

LEVD_TEST_CASES = [
    ("kitten", "sitting", 3),
    ("saturday", "sunday", 3),
    ("", "", 0),
    ("a", "", 1),
    ("", "a", 1),
    ("abc", "abc", 0),
    ("book", "back", 2),
    ("test", "text", 1),
    ("flaw", "lawn", 2),
    ("intention", "execution", 5),
]

# --- CPED Backend Correctness Tests ---


@pytest.mark.parametrize("X, Y, expected", CPED_TEST_CASES)
@pytest.mark.parametrize("backend", ["python", "cpp", "c++"])
def test_cped_correctness(X, Y, expected, backend):
    assert repmetric.cped(X, Y, backend=backend) == expected


@pytest.mark.parametrize("backend", ["python", "cpp", "c++"])
def test_cped_matrix_correctness(backend):
    sequences = ["a", "ab", "abc"]
    expected_matrix = np.array([[0, 1, 1], [1, 0, 1], [2, 1, 0]])
    dist_matrix = repmetric.cped_matrix(sequences, backend=backend)
    # Corrected expected matrix based on re-evaluating cped logic
    expected_matrix = np.array([[0, 1, 2], [1, 0, 1], [1, 1, 0]])
    np.testing.assert_array_equal(dist_matrix, expected_matrix)


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_bicped_improvement(backend):
    baseline = repmetric.edit_distance(
        "", "aaaba", distance_type="cped", backend=backend
    )
    improved = repmetric.edit_distance(
        "", "aaaba", distance_type="bicped", backend=backend
    )
    assert baseline >= improved
    assert improved == 4


# --- Levenshtein Backend Correctness Tests ---


@pytest.mark.parametrize("s1, s2, expected", LEVD_TEST_CASES)
@pytest.mark.parametrize("backend", ["python", "cpp", "c++"])
def test_levd_correctness(s1, s2, expected, backend):
    assert repmetric.levd(s1, s2, backend=backend) == expected


@pytest.mark.parametrize("backend", ["python", "cpp", "c++"])
def test_levd_matrix_correctness(backend):
    sequences = ["a", "ab", "abc"]
    expected_matrix = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    dist_matrix = repmetric.levd_matrix(sequences, backend=backend)
    np.testing.assert_array_equal(dist_matrix, expected_matrix)


# --- High-Level API Tests ---


@pytest.mark.parametrize("s1, s2, expected", LEVD_TEST_CASES)
def test_edit_distance_levd(s1, s2, expected):
    assert repmetric.edit_distance(s1, s2, distance_type="levd") == expected


@pytest.mark.parametrize("X, Y, expected", CPED_TEST_CASES)
def test_edit_distance_cped(X, Y, expected):
    assert repmetric.edit_distance(X, Y, distance_type="cped") == expected


def test_edit_distance_matrix():
    sequences = ["a", "ab", "abc"]
    expected_levd = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    expected_cped = np.array([[0, 1, 2], [1, 0, 1], [1, 1, 0]])

    # Test parallel C++ implementation
    np.testing.assert_array_equal(
        repmetric.edit_distance(
            sequences, distance_type="levd", backend="cpp", parallel=True
        ),
        expected_levd,
    )
    np.testing.assert_array_equal(
        repmetric.edit_distance(
            sequences, distance_type="cped", backend="cpp", parallel=True
        ),
        expected_cped,
    )

    # Test sequential C++ implementation
    np.testing.assert_array_equal(
        repmetric.edit_distance(
            sequences, distance_type="levd", backend="cpp", parallel=False
        ),
        expected_levd,
    )
    np.testing.assert_array_equal(
        repmetric.edit_distance(
            sequences, distance_type="cped", backend="cpp", parallel=False
        ),
        expected_cped,
    )

    # Test python implementation
    np.testing.assert_array_equal(
        repmetric.edit_distance(sequences, distance_type="levd", backend="python"),
        expected_levd,
    )
    np.testing.assert_array_equal(
        repmetric.edit_distance(sequences, distance_type="cped", backend="python"),
        expected_cped,
    )
    np.testing.assert_array_equal(
        repmetric.edit_distance(sequences, distance_type="bicped", backend="python"),
        expected_cped,
    )


def test_edit_distance_invalid_args():
    with pytest.raises(ValueError):
        repmetric.edit_distance("a", "b", distance_type="invalid")
    with pytest.raises(ValueError):
        repmetric.edit_distance(["a"], "b")
    with pytest.raises(TypeError):
        repmetric.edit_distance("a")  # Missing b


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_edit_distance_bicped(backend):
    assert (
        repmetric.edit_distance("", "aaaba", distance_type="bicped", backend=backend)
        == 4
    )


def test_edit_distance_bicped_cpp_backend():
    # Test that the C++ backend for bicped can be called without raising an error.
    # Correctness is checked in other tests.
    try:
        repmetric.edit_distance("a", "b", distance_type="bicped", backend="cpp")
        repmetric.edit_distance(["a", "b"], distance_type="bicped", backend="cpp")
    except ValueError:
        pytest.fail("Calling bicped with cpp backend should not raise a ValueError.")


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_bicped_function(backend):
    assert repmetric.bicped("", "aaaba", backend=backend) == 4


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_bicped_matrix_function(backend):
    sequences = ["a", "ab", "abc"]
    expected = np.array([[0, 1, 2], [1, 0, 1], [1, 1, 0]])
    np.testing.assert_array_equal(
        repmetric.bicped_matrix(sequences, backend=backend), expected
    )


# --- Fallback Mechanism Tests ---


@patch("repmetric.api.CPP_AVAILABLE", False)
def test_fallback_to_python():
    with patch("repmetric.api._calculate_cped_py") as mock_cped_py:
        mock_cped_py.return_value = 123
        result = repmetric.cped("a", "b", backend="cpp")
        mock_cped_py.assert_called_once_with("a", "b")
        assert result == 123

    with patch("repmetric.api._calculate_levd_py") as mock_levd_py:
        mock_levd_py.return_value = 456
        result = repmetric.levd("x", "y", backend="c++")
        mock_levd_py.assert_called_once_with("x", "y")
        assert result == 456


# --- Axiom/Property Tests ---


def test_non_negativity():
    assert repmetric.cped("abc", "def") >= 0
    assert repmetric.levd("abc", "def") >= 0


def test_identity_of_indiscernibles():
    assert repmetric.cped("abc", "abc") == 0
    assert repmetric.levd("abc", "abc") == 0
    assert repmetric.cped("abc", "abd") != 0
    assert repmetric.levd("abc", "abd") != 0


def test_symmetry_levd():
    assert repmetric.levd("apple", "apply") == repmetric.levd("apply", "apple")


def test_asymmetry_cped():
    assert repmetric.cped("x", "abab") != repmetric.cped("abab", "x")


def test_triangle_inequality():
    # Levenshtein
    x, y, z = "book", "boook", "booom"
    assert repmetric.levd(x, z) <= repmetric.levd(x, y) + repmetric.levd(y, z)
    # CPED
    x, y, z = "ab", "abc", "abcd"
    assert repmetric.cped(x, z) <= repmetric.cped(x, y) + repmetric.cped(y, z)


@pytest.mark.parametrize("X, Y, _", CPED_TEST_CASES)
def test_cped_le_levenshtein(X, Y, _):
    assert repmetric.cped(X, Y) <= repmetric.levd(X, Y)
