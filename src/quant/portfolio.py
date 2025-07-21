"""Portfolio class"""

from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import backtest, constants, utils


class Portfolio:
    """A class to represent a group of asset returns. The assets can be assigned
    weights and rebalanced to create a portfolio for backtesting.

    The Portfolio class provides functionality for:
    - Managing asset returns and weights
    - Calculating portfolio returns with rebalancing
    - Optimizing portfolio weights using various strategies
    - Performance analysis and benchmarking
    """

    def __init__(
        self,
        returns: pd.DataFrame | list,  # as of period end
        weights: pd.DataFrame | pd.Series | list | None = None,  # as of period start
        rebalance_freq: str = "M",
        benchmark: pd.DataFrame | str | None = None,
    ):
        """Initialize a Portfolio object.

        Args:
            returns: DataFrame of asset returns or list of ticker symbols
            weights: Portfolio weights (DataFrame, Series, list, or None for no weights)
            rebalance_freq: Frequency of rebalancing ('D', 'W', 'M', 'Q', 'Y', 'custom')
            benchmark: Optional benchmark returns for performance comparison
        """
        if isinstance(returns, pd.DataFrame):
            self.returns = returns
        elif isinstance(returns, list):
            self.returns = (
                pd.DataFrame().quant.ticker(returns).quant.palign().quant.to_returns()
            )
        else:
            raise ValueError(
                "returns must be a DataFrame of returns or a list of tickers"
            )

        self.assets = self.returns.columns.to_list()

        self._validate_rebalance_freq(rebalance_freq)
        self.rebalance_freq = rebalance_freq

        self.set_weights(weights, rebalance_freq)

        self.benchmark = benchmark

    def __repr__(self):
        """Return a string representation of the portfolio."""
        # Basic portfolio info
        info = [
            f"Portfolio(assets={len(self.assets)})",
            f"Period: {self.returns.index[0].strftime('%Y-%m-%d')} to {self.returns.index[-1].strftime('%Y-%m-%d')}",
            f"Rebalance: {self.rebalance_freq}",
        ]

        if len(self.assets) <= 5:
            info.append(f"Assets: {', '.join(self.assets)}")
        else:
            info.append(
                f"Assets: {', '.join(self.assets[:3])}... and {len(self.assets) - 3} more"
            )

        # Add portfolio statistics if weights are set
        if self.weights is not None:
            latest_weights = self.weights.iloc[-1]
            top_holdings = latest_weights.nlargest(3)
            holdings_str = ", ".join(
                [f"{asset}: {weight:.1%}" for asset, weight in top_holdings.items()]
            )
            info.append(f"Top Holdings: {holdings_str}")

        return "\n".join(info)

    def copy(self):
        """Create a deep copy of the portfolio."""
        return deepcopy(self)

    def _calc_effective_weights(self):
        """Calculate weights that align with returns frequency using self.weights accounting
        for intra rebalancing weight fluctuations due to returns.

        This method handles the drift in portfolio weights between rebalancing dates
        due to the different performance of assets. It creates a high-frequency
        weight series that shows how weights evolve between rebalancing points.

        Note: Weights must be set before calling this method.
        """
        if self.weights is None:
            raise ValueError(
                "Weights are not set. Please provide weights to calculate portfolio returns."
            )
        rebalance = utils.align_index_forward(self.weights, self.returns)
        num_rebalance = len(rebalance)

        self.eff_weights = pd.DataFrame(columns=self.assets, index=self.returns.index)
        updated_weight = rebalance.iloc[0]
        lf_ptr = 0
        for hf_ptr in range(len(self.eff_weights)):
            if rebalance.index[lf_ptr] == self.eff_weights.index[hf_ptr]:
                self.eff_weights.iloc[hf_ptr] = rebalance.iloc[lf_ptr]
                if lf_ptr < num_rebalance - 1:
                    lf_ptr += 1
            else:
                self.eff_weights.iloc[hf_ptr] = updated_weight

            # Update weights for next period based on asset returns
            # This simulates how weights drift between rebalancing dates
            updated_weight = self.eff_weights.iloc[hf_ptr] * (
                1 + self.returns.iloc[hf_ptr]
            )
            updated_weight /= updated_weight.sum()

        return None

    def _calc_portfolio_returns(self):
        """Calculate portfolio returns using effective weights and asset returns."""
        self._calc_effective_weights()
        self.portfolio_returns = (
            (self.eff_weights * self.returns).sum(axis=1).to_frame("portfolio")
        )
        return None

    def set_benchmark(self, benchmark):
        """Set a benchmark for performance comparison.

        Args:
            benchmark: DataFrame containing benchmark returns

        Raises:
            ValueError: If benchmark is not a single asset or a DataFrame
        """
        if isinstance(benchmark, str):
            benchmark = pd.DataFrame.quant.ticker(benchmark).quant.to_returns()
        elif isinstance(benchmark, pd.DataFrame):
            if benchmark.shape[1] != 1:
                raise ValueError("Benchmark must be a single asset")
        else:
            raise ValueError("Benchmark must be a single asset or a DataFrame")

        self.benchmark = benchmark

    def set_weights(
        self,
        weights: pd.DataFrame | pd.Series | list | None,
        rebalance_freq=None,
    ):
        """Set weights across assets using rebalance frequency. When using custom rebalance frequency,
        weights are assigned with no processing.

        Args:
            weights: Portfolio weights (DataFrame, Series, list, or None to clear weights)
            rebalance_freq: Frequency of rebalancing (uses self.rebalance_freq if None)
        """
        if rebalance_freq is None:
            rebalance_freq = self.rebalance_freq

        self._validate_rebalance_freq(rebalance_freq)

        match weights:
            case None:
                self.weights = None
            case pd.DataFrame():
                if not self.returns.columns.equals(weights.columns):
                    raise ValueError("returns and weights must have same columns")
                if rebalance_freq == "custom":
                    self.weights = weights
                else:
                    self.weights = (
                        weights.resample(constants.SAMPLING[rebalance_freq])
                        .last()
                        .loc[self.returns.index[0] : self.returns.index[-1]]
                    )
            case list() | pd.Series():
                if len(weights) != len(self.assets):
                    raise ValueError("weights must have same length as assets")
                self.weights = pd.DataFrame(
                    np.ones((len(self.returns), len(self.assets))),
                    index=self.returns.index,
                    columns=self.assets,
                ).mul(weights, axis=1)  # type: ignore
                if rebalance_freq != "custom":
                    self.weights = (
                        self.weights.resample(constants.SAMPLING[rebalance_freq])
                        .last()
                        .loc[: self.returns.index[-1]]
                    )
            case _:
                raise TypeError("Invalid weights type")

        if weights is not None:
            self._calc_portfolio_returns()

        return None

    def _validate_rebalance_freq(self, rebalance_freq):
        """Validate that the rebalancing frequency is supported.

        Args:
            rebalance_freq: Frequency string to validate
        """
        if rebalance_freq not in constants.SAMPLING and rebalance_freq != "custom":
            raise ValueError(
                f"Invalid rebalance frequency! Choose from {list(constants.SAMPLING.keys()) + ['custom']}"
            )
        return None

    def update_rebalance_freq(self, rebalance_freq: str):
        """Update the rebalancing frequency and recalculate portfolio returns.

        Args:
            rebalance_freq: New rebalancing frequency
        """
        if self.weights is None:
            raise ValueError("Weights are not set")
        self._validate_rebalance_freq(rebalance_freq)
        self.rebalance_freq = rebalance_freq
        self.set_weights(self.weights, rebalance_freq)
        return None

    def perf_summary(self, benchmark=None, yr=constants.YEAR_BY["day"]):
        """Generate performance summary table for the portfolio.

        Args:
            benchmark: Optional benchmark for comparison (uses self.benchmark if None)
            yr: Number of periods in a year for annualization

        Returns:
            Styled performance summary table
        """
        if benchmark is None:
            benchmark = self.benchmark
        return backtest.perf_summary_table(self.portfolio_returns, bmk=benchmark, yr=yr)

    def _get_default_optimization_constraints(self, n_assets):
        """Get default optimization constraints for portfolio weights.

        Args:
            n_assets: Number of assets in the portfolio

        Returns:
            tuple: (bounds, constraints) for scipy.minimize
                  Use None for unconstrained/unbounded optimization
        """
        bounds = [(0, 1) for _ in range(n_assets)]  # weights between 0 and 1
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        return bounds, constraints

    def mvo_weights(
        self,
        mu=None,
        sigma=None,
        rebalance_freq=None,
        bounds="default",
        constraints="default",
    ):
        """Calculate weights for a portfolio that maximizes the Sharpe ratio using mean-variance optimization.

        This implements mean-variance optimization to maximize the Sharpe ratio
        using scipy's optimization capabilities.

        Args:
            mu: Expected returns (uses historical means if None)
            sigma: Covariance matrix (uses historical covariance if None)
            rebalance_freq: Frequency for returns aggregation (uses self.rebalance_freq if None)
            bounds: List of (min, max) tuples for each asset weight, "default" for default bounds,
                or None for unbounded
            constraints: List of constraint dictionaries for scipy.minimize, "default" for default
                constraints, or None for unconstrained

        Returns:
            pd.Series: Optimal weights for each asset
        """
        if rebalance_freq is None:
            rebalance_freq = self.rebalance_freq

        if mu is None:
            if rebalance_freq == "custom":
                mu = self.returns.mean()
            else:
                mu = self.returns.quant.agg_returns(rebalance_freq).mean()

        if sigma is None:
            if rebalance_freq == "custom":
                sigma = self.returns.cov()
            else:
                sigma = self.returns.quant.agg_returns(rebalance_freq).cov()

        n_assets = len(self.assets)

        def neg_sharpe_ratio(weights):
            """Negative of the Sharpe ratio for minimization.

            This function calculates the negative of the Sharpe ratio
            of the portfolio, which we minimize to find optimal weights.
            """
            portfolio_return = np.sum(weights * mu)
            portfolio_vol = np.sqrt(weights.T @ sigma @ weights)

            # Avoid division by zero
            if portfolio_vol == 0:
                return 1e6

            sharpe_ratio = portfolio_return / portfolio_vol
            return -sharpe_ratio

        # Apply defaults if requested
        if bounds == "default" or constraints == "default":
            default_bounds, default_constraints = (
                self._get_default_optimization_constraints(n_assets)
            )
            bounds = default_bounds if bounds == "default" else bounds
            constraints = (
                default_constraints if constraints == "default" else constraints
            )

        # Initial guess: equal weights
        initial_weights = np.array([1 / n_assets] * n_assets)

        # Optimize using scipy's SLSQP method
        result = minimize(
            neg_sharpe_ratio,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-8, "disp": False},
        )

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        weights = pd.Series(result.x, index=self.assets)

        return weights

    def kelly_weights(
        self,
        rebalance_freq=None,
        bounds="default",
        constraints="default",
    ):
        """Calculate optimal weights using the Kelly criterion by maximizing log returns.

        The Kelly criterion maximizes the expected logarithm of wealth, which
        is equivalent to maximizing the geometric mean return.

        Args:
            rebalance_freq: Optional frequency for returns aggregation
            bounds: List of (min, max) tuples for each asset weight, "default" for default bounds,
                or None for unbounded
            constraints: List of constraint dictionaries for scipy.minimize, "default" for default
                constraints, or None for unconstrained

        Returns:
            pd.Series: Optimal weights for each asset
        """
        if rebalance_freq is None:
            rebalance_freq = self.rebalance_freq

        if rebalance_freq == "custom":
            returns = self.returns
        else:
            returns = self.returns.quant.agg_returns(rebalance_freq)

        n_assets = len(self.assets)

        def neg_log_return(weights):
            """Negative of the expected log return for minimization."""
            portfolio_returns = returns.dot(weights)
            return -np.mean(np.log(1 + portfolio_returns))

        # Apply defaults if requested
        if bounds == "default" or constraints == "default":
            default_bounds, default_constraints = (
                self._get_default_optimization_constraints(n_assets)
            )
            bounds = default_bounds if bounds == "default" else bounds
            constraints = (
                default_constraints if constraints == "default" else constraints
            )

        # Initial guess: equal weights
        initial_weights = np.array([1 / n_assets] * n_assets)

        # Optimize using scipy's SLSQP method
        result = minimize(
            neg_log_return,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-8, "disp": False},
        )

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        weights = pd.Series(result.x, index=self.assets)

        return weights

    def risk_budget_weights(self, rebalance_freq=None, risk_budget=None):
        """Calculate risk budget weights where each asset's risk contribution matches a target risk budget.

        Risk budget optimization aims to achieve a specific risk contribution from each asset
        according to the provided risk budget. By default, equal risk contribution is targeted.

        Args:
            rebalance_freq: Optional frequency for returns aggregation
            risk_budget: Array of target risk contributions for each asset (must sum to 1).
                        If None, equal risk contribution is used (1/n_assets for each asset)

        Returns:
            pd.Series: Risk budget weights for each asset
        """
        if rebalance_freq is None:
            rebalance_freq = self.rebalance_freq

        # Get returns at the specified frequency
        if rebalance_freq == "custom":
            returns = self.returns
        else:
            returns = self.returns.quant.agg_returns(rebalance_freq)

        n_assets = len(self.assets)

        # Set default risk budget to equal risk contribution
        if risk_budget is None:
            risk_budget = np.array([1 / n_assets] * n_assets)
        else:
            risk_budget = np.array(risk_budget)
            # Validate that risk budget sums to 1
            if not np.isclose(np.sum(risk_budget), 1.0, atol=1e-6):
                raise ValueError("Risk budget must sum to 1")
            if len(risk_budget) != n_assets:
                raise ValueError(
                    f"Risk budget length ({len(risk_budget)}) must match number of assets ({n_assets})"
                )

        # Calculate covariance matrix
        sigma = returns.cov()

        def risk_budget_objective(weights):
            """Objective function to minimize the difference between actual and target risk contributions.

            This function calculates the sum of squared differences between actual risk contributions
            and the target risk budget. When minimized, actual risk contributions will match the target.
            """
            weights = weights / np.sum(weights)

            portfolio_var = weights.T @ sigma @ weights

            risk_contributions = (sigma @ weights) * weights / portfolio_var

            # Calculate sum of squared differences from target risk budget
            error = np.sum((risk_contributions - risk_budget) ** 2)

            return error

        # Set up constraints and bounds
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        bounds = [(0, 1) for _ in range(n_assets)]  # weights between 0 and 1

        initial_weights = np.array([1 / n_assets] * n_assets)

        # Optimize using scipy's SLSQP method
        result = minimize(
            risk_budget_objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-8, "disp": False},
        )

        if not result.success:
            raise ValueError(f"Risk budget optimization failed: {result.message}")

        weights = pd.Series(result.x, index=self.assets)

        return weights
