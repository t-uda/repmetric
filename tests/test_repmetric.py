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

BICPED_TEST_CASES = [
    ("", "", 0),
    ("", "aaaba", 4),
    ("abcdef", "abcdef", 0),
    ("abc", "ac", 1),
    ("ab", "abab", 1),
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


@pytest.mark.parametrize("X, Y, expected", BICPED_TEST_CASES)
@pytest.mark.parametrize("backend", ["python", "cpp", "c++"])
def test_bicped_correctness(X, Y, expected, backend):
    assert repmetric.bicped(X, Y, backend=backend) == expected


def test_bicped_improvement():
    baseline = repmetric.edit_distance(
        "", "aaaba", distance_type="cped", backend="python"
    )
    improved_python = repmetric.edit_distance(
        "", "aaaba", distance_type="bicped", backend="python"
    )
    improved_cpp = repmetric.edit_distance(
        "", "aaaba", distance_type="bicped", backend="cpp"
    )
    assert baseline >= improved_python
    assert improved_python == 4
    assert improved_cpp == 4


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
        repmetric.edit_distance(
            sequences, distance_type="bicped", backend="cpp", parallel=True
        ),
        expected_cped,
    )
    np.testing.assert_array_equal(
        repmetric.edit_distance(
            sequences, distance_type="bicped", backend="cpp", parallel=False
        ),
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


@pytest.mark.parametrize("backend", ["python", "cpp", "c++"])
def test_edit_distance_bicped(backend):
    assert (
        repmetric.edit_distance("", "aaaba", distance_type="bicped", backend=backend)
        == 4
    )


def test_edit_distance_bicped_invalid_backend():
    with pytest.raises(ValueError):
        repmetric.edit_distance("a", "b", distance_type="bicped", backend="rust")
    with pytest.raises(ValueError):
        repmetric.edit_distance(["a", "b"], distance_type="bicped", backend="rust")


def test_bicped_function():
    assert repmetric.bicped("", "aaaba", backend="python") == 4
    assert repmetric.bicped("", "aaaba", backend="cpp") == 4


def test_bicped_matrix_function():
    sequences = ["a", "ab", "abc"]
    expected = np.array([[0, 1, 2], [1, 0, 1], [1, 1, 0]])
    np.testing.assert_array_equal(
        repmetric.bicped_matrix(sequences, backend="python"), expected
    )
    np.testing.assert_array_equal(
        repmetric.bicped_matrix(sequences, backend="cpp", parallel=True), expected
    )
    np.testing.assert_array_equal(
        repmetric.bicped_matrix(sequences, backend="cpp", parallel=False), expected
    )
    np.testing.assert_array_equal(
        repmetric.bicped_matrix(sequences, backend="c++", parallel=True), expected
    )


# --- Fallback Mechanism Tests ---


@patch("repmetric.api.CPP_AVAILABLE", False)
def test_fallback_to_python():
    with patch("repmetric.api._calculate_cped_py") as mock_cped_py:
        mock_cped_py.return_value = 123
        result = repmetric.cped("a", "b", backend="cpp")
        mock_cped_py.assert_called_once_with("a", "b")
        assert result == 123

    with patch("repmetric.api._calculate_bicped_py") as mock_bicped_py:
        mock_bicped_py.return_value = 789
        result = repmetric.bicped("a", "b", backend="cpp")
        mock_bicped_py.assert_called_once_with("a", "b")
        assert result == 789

    with patch(
        "repmetric.api._calculate_bicped_distance_matrix_py"
    ) as mock_bicped_matrix_py:
        mock_bicped_matrix_py.return_value = np.array([[0, 1], [1, 0]])
        result = repmetric.bicped_matrix(["a", "b"], backend="cpp")
        mock_bicped_matrix_py.assert_called_once_with(["a", "b"])
        np.testing.assert_array_equal(result, mock_bicped_matrix_py.return_value)

    with patch("repmetric.api._calculate_levd_py") as mock_levd_py:
        mock_levd_py.return_value = 456
        result = repmetric.levd("x", "y", backend="c++")
        mock_levd_py.assert_called_once_with("x", "y")
        assert result == 456

    with patch(
        "repmetric.api._calculate_cped_distance_matrix_py"
    ) as mock_cped_matrix_py:
        mock_cped_matrix_py.return_value = np.array([[0, 2], [1, 0]])
        sequences = ["foo", "bar"]
        result = repmetric.cped_matrix(sequences, backend="cpp")
        mock_cped_matrix_py.assert_called_once_with(sequences)
        np.testing.assert_array_equal(result, mock_cped_matrix_py.return_value)

    with patch(
        "repmetric.api._calculate_levd_distance_matrix_py"
    ) as mock_levd_matrix_py:
        mock_levd_matrix_py.return_value = np.array([[0, 3], [3, 0]])
        sequences = ["alpha", "beta"]
        result = repmetric.levd_matrix(sequences, backend="cpp")
        mock_levd_matrix_py.assert_called_once_with(sequences)
        np.testing.assert_array_equal(result, mock_levd_matrix_py.return_value)


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
