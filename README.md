# **Quant Finance**

A collection of tools, scripts, and resources for quantitative finance, covering financial modeling, trading strategies, and data analysis.

## **Table of Contents**

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## **Introduction**

Quantitative finance leverages mathematical models, statistical techniques, and computational tools to analyze financial markets and develop trading strategies. This repository provides a structured collection of scripts and resources to support research, modeling, and algorithmic trading. The backbone of this project is `quant` accessor that is built on top of `pandas.DataFrame` to add functionalities relevant to financial data analysis. The `backtest` module helps with performance backtesting and risk decomposition. The `Portfolio` class is a customizable framework to create portfolios that account for rebalancing.

## **Installation**

To set up the project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nakulrandad/Quant-Finance.git
    cd Quant-Finance
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    uv venv
    source .venv/bin/activate
    ```

3.  **Install the package:**

    *   **Standard Installation:**
        ```bash
        uv pip install .
        ```

    *   **For Development (Editable Mode):**
        ```bash
        uv pip install -e .[dev]
        ```

## **Usage**
The `analysis` folder contains some sample analysis to showcase some of the functionalities of quant package. This package is actively under development and any feedback and/or contribution would be highly appreciated.

## **License**

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
