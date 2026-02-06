# repmetric

## Project overview
Repmetric provides fast implementations of edit distance algorithms with interchangeable backends. It supports multiple distance types—CPED, BICPed, and Levenshtein—to help compare sequences efficiently for research and production workloads. The library exposes a simple Python API while offering a high-performance C++ extension for critical paths.

> **Status:** Repmetric is an alpha-stage library developed for academic research purposes. It is not published on PyPI or any other public package index.

## Install from the repository
Until a public release is prepared, install Repmetric directly from the GitHub repository:

```bash
pip install git+https://github.com/t-uda/repmetric
```

## Development workflow
The following instructions target contributors who need the full development environment and tooling.

### Install development dependencies
Repmetric uses [Poetry](https://python-poetry.org/) for dependency management. Set up the virtual environment with:

```bash
poetry install
```

This creates (or reuses) a virtual environment configured with all required runtime and development packages.

### Compile the C++ extension
Some distance computations can leverage a compiled backend for additional speed. After installing dependencies, build the extension in-place:

```bash
poetry run python setup.py build_ext --inplace
```

Re-run this command whenever the C++ sources change (for example, after editing `src/cped.cpp`).

### Run tests and linters
Use Poetry to execute the provided quality checks:

```bash
poetry run pytest
poetry run ruff check .
poetry run black src tests notebooks
poetry run mypy src tests
```

All commands should complete without errors before committing changes.

## Usage examples
The package offers both pure Python (`python`) and compiled (`cpp`/`c++`) backends. You can specify the distance type (`cped`, `levd`/`levenshtein`, or `bicped`) and backend when calling `repmetric.edit_distance`:

```python
from repmetric import edit_distance

# Compute CPED distance using the default Python backend
cost = edit_distance("kitten", "sitting", distance_type="cped", backend="python")

# Switch to the C++ backend for higher performance and select a different metric
cost_cpp = edit_distance("kitten", "sitting", distance_type="levenshtein", backend="cpp")
```

Consult the API reference and tests for additional usage patterns, including the BICPed distance configuration.
