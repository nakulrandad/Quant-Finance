# Gemini Project Context: Quant-Finance

This document provides a comprehensive overview of the `Quant-Finance` project for the Gemini AI assistant. It outlines the project's purpose, architecture, and development conventions to ensure effective and context-aware collaboration.

## Project Overview

`Quant-Finance` is a Python-based library for quantitative finance analysis. It provides a suite of tools for financial modeling, trading strategy development, backtesting, and portfolio management.

The core of the library is a custom `.quant` accessor for pandas DataFrames and Series, which extends their functionality with a rich set of methods for financial calculations and data retrieval.

**Key Features:**

*   **Financial Analysis:** Calculate metrics like annualized returns, volatility, Sharpe ratio, beta, alpha, and more.
*   **Backtesting:** Evaluate the performance of trading strategies with detailed reports and visualizations.
*   **Portfolio Management:** Create, manage, and rebalance portfolios with various weight optimization strategies (Mean-Variance Optimization, Kelly Criterion, Risk Budgeting).
*   **Data Retrieval:** Fetch financial data from sources like Yahoo Finance, FRED (Federal Reserve Economic Data), and Indian Mutual Funds.

**Technologies:**

*   **Language:** Python 3.11+
*   **Core Libraries:** pandas, numpy, scipy, matplotlib, seaborn, statsmodels
*   **Package Management:** `uv`
*   **Testing:** `pytest`
*   **Linting & Formatting:** `ruff`, `isort`
*   **Type Checking:** `mypy`

## Building and Running

### Installation

The project uses `uv` for package management.

1.  **Create a virtual environment:**
    ```bash
    uv venv
    source .venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    uv pip install -e .[dev]
    ```

### Running Tests

The project uses `pytest` for testing. To run the test suite:

```bash
pytest
```

### Usage

The library is intended to be used in Python scripts or Jupyter notebooks. The `analysis` directory contains example notebooks demonstrating the library's functionality.

**Example:**

```python
import pandas as pd
import quant

# Fetch historical data
prices = pd.DataFrame().quant.ticker("SPY")

# Calculate returns
returns = prices.quant.to_returns()

# Calculate annualized volatility
volatility = returns.quant.return_vol()

print(volatility)
```

## Development Conventions

*   **Coding Style:** The project follows the `ruff` code style. Code is automatically formatted using `ruff-format` via pre-commit hooks.
*   **Linting:** `ruff` is used for linting to ensure code quality.
*   **Type Hinting:** The project uses type hints, and `mypy` is used for static type checking.
*   **Pre-commit Hooks:** The repository is configured with pre-commit hooks to automatically format code, sort imports, and run linters before each commit. The configuration is in `.pre-commit-config.yaml`.
*   **Modularity:** The codebase is organized into modules with specific responsibilities (e.g., `backtest.py`, `portfolio.py`).

## Key Files

*   `pyproject.toml`: Defines project metadata, dependencies, and build configuration.
*   `src/quant/quant_accessor.py`: The core of the library, defining the `.quant` accessor for pandas objects.
*   `src/quant/backtest.py`: Contains functions for backtesting and performance analysis.
*   `src/quant/portfolio.py`: Implements the `Portfolio` class for managing and optimizing investment portfolios.
*   `src/quant/api.py`: Handles data retrieval from external APIs (FRED, Yahoo Finance, etc.).
*   `tests/`: Contains the test suite for the project.
*   `.pre-commit-config.yaml`: Configuration for the pre-commit hooks.
*   `mypy.ini`: Configuration for the `mypy` static type checker.
*   `analysis/`: Contains Jupyter notebooks with example usage of the library.
*   `README.md`: The main project README file.
*   `GEMINI.md`: This file, providing context for the Gemini AI assistant.
