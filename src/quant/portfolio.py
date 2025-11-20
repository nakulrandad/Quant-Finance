"""Portfolio class"""

from copy import deepcopy

import numpy as np
import pandas as pd

from . import backtest, constants, optimizers, utils


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

    def mvo_weights(
        self,
        mu=None,
        sigma=None,
        rebalance_freq=None,
        bounds="default",
        constraints="default",
    ):
        """Calculate weights for a portfolio that maximizes the Sharpe ratio using mean-variance optimization.

        Wrapper around MeanVarianceOptimizer.
        """
        if rebalance_freq is None:
            rebalance_freq = self.rebalance_freq

        # Aggregate returns if needed
        if rebalance_freq == "custom":
            agg_returns = self.returns
        else:
            agg_returns = self.returns.quant.agg_returns(rebalance_freq)

        optimizer = optimizers.MeanVarianceOptimizer()
        return optimizer.optimize(
            agg_returns,
            mu=mu,
            sigma=sigma,
            bounds=bounds,
            constraints=constraints,
        )

    def kelly_weights(
        self,
        rebalance_freq=None,
        bounds="default",
        constraints="default",
    ):
        """Calculate optimal weights using the Kelly criterion.

        Wrapper around KellyOptimizer.
        """
        if rebalance_freq is None:
            rebalance_freq = self.rebalance_freq

        # Aggregate returns if needed
        if rebalance_freq == "custom":
            agg_returns = self.returns
        else:
            agg_returns = self.returns.quant.agg_returns(rebalance_freq)

        optimizer = optimizers.KellyOptimizer()
        return optimizer.optimize(
            agg_returns,
            bounds=bounds,
            constraints=constraints,
        )

    def risk_budget_weights(self, rebalance_freq=None, risk_budget=None):
        """Calculate risk budget weights.

        Wrapper around RiskBudgetOptimizer.
        """
        if rebalance_freq is None:
            rebalance_freq = self.rebalance_freq

        # Aggregate returns if needed
        if rebalance_freq == "custom":
            agg_returns = self.returns
        else:
            agg_returns = self.returns.quant.agg_returns(rebalance_freq)

        optimizer = optimizers.RiskBudgetOptimizer()
        return optimizer.optimize(
            agg_returns,
            risk_budget=risk_budget,
        )
