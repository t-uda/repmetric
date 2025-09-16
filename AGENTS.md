This file provides instructions for AI agents working on this repository.

## Development Workflow

### 1. Initial Environment Setup

This project includes a C++ extension. To set up your environment for development, you must install the Python dependencies and then compile the extension.

Run these commands once after cloning the repository:

```bash
# 1. Install Python dependencies
poetry install

# 2. Compile the C++ extension for local development
poetry run python setup.py build_ext --inplace
```

**Note:** If you make any changes to the C++ source code (`src/cped.cpp`), you must re-run the `build_ext` command to see your changes.

### 2. Running Tests and Linters

Before pushing your changes, ensure all checks pass:

```bash
poetry run pytest
poetry run ruff check .
poetry run black src tests notebooks
poetry run mypy src tests
```

### 3. Executing Notebooks

The notebooks in this repository can be executed using `papermill` to ensure reproducibility and to run them with different parameters.

First, you need to register the project's virtual environment as a Jupyter kernel. Run this command once:

```bash
poetry run python -m ipykernel install --user --name=repmetric --display-name "Python (repmetric)"
```

Then, to execute a notebook, use the following command structure:

```bash
poetry run papermill notebooks/input.ipynb notebooks/output.ipynb -p PARAM_NAME PARAM_VALUE
```

For example, to run the performance comparison:
```bash
poetry run papermill notebooks/performance_comparison.ipynb notebooks/performance_comparison_output.ipynb
```

**Note on Committing Notebooks:** As a general rule, do not commit the execution results of notebooks. Notebooks should only be executed for verification purposes, and the output cells should be cleared before committing.

## Language and Style

*   **Coding Style:** All Python code should adhere to the [PEP 8 style guide](https://peps.python.org/pep-0008/).
*   **Docstrings:** Docstrings must be written in **English**. They can include simple mathematical expressions where appropriate.
*   **Comments:** Code comments should be short and simple. Do not add verbose trivial comments.
