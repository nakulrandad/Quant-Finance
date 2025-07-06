"""Portfolio class"""

from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import backtest, constants, utils


class Portfolio:
    """A class to represent a group of asset returns. The assets can be assigned
    weights and rebalanced to create a portfolio for backtesting.
    """

    def __init__(
        self,
        returns: Union[pd.DataFrame, list],  # as of period end
        weights: Union[
            pd.DataFrame, pd.Series, list, None
        ] = None,  # as of period start
        rebalance_freq: str = "M",
        benchmark: Union[pd.DataFrame, None] = None,
    ):
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
        
        # Add assets if there are few enough
        if len(self.assets) <= 5:
            info.append(f"Assets: {', '.join(self.assets)}")
        else:
            info.append(f"Assets: {', '.join(self.assets[:3])}... and {len(self.assets)-3} more")
            
        # Add portfolio statistics if weights are set
        if self.weights is not None:
            latest_weights = self.weights.iloc[-1]
            top_holdings = latest_weights.nlargest(3)
            holdings_str = ", ".join([f"{asset}: {weight:.1%}" for asset, weight in top_holdings.items()])
            info.append(f"Top Holdings: {holdings_str}")
        
        return "\n".join(info)

    def copy(self):
        return deepcopy(self)

    def _calc_effective_weights(self):
        """Calculate weights that align with returns frequency using self.weights accounting
        for intra rebalancing weight fluctuations due to returns.
        """
        if self.weights is None:
            raise ValueError(
                "Weights are not set. Please provide weights to calculate portfolio returns."
            )
        rebalance = utils.fix_low_freq_index(self.weights, self.returns)
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

            updated_weight = self.eff_weights.iloc[hf_ptr] * (
                1 + self.returns.iloc[hf_ptr]
            )
            updated_weight /= updated_weight.sum()

        return None

    def _calc_portfolio_returns(self):
        self._calc_effective_weights()
        self.portfolio_returns = (
            (self.eff_weights * self.returns).sum(axis=1).to_frame("portfolio")
        )
        return None

    def set_benchmark(self, benchmark):
        self.benchmark = benchmark

    def set_weights(
        self,
        weights: Union[pd.DataFrame, pd.Series, list, None],
        rebalance_freq=None,
    ):
        """Set weights across assets using rebalance frequency. When using custom rebalance frequency,
        weights are assigned with no processing.
        """
        if rebalance_freq is None:
            rebalance_freq = self.rebalance_freq

        self._validate_rebalance_freq(rebalance_freq)

        if weights is not None:
            if isinstance(weights, pd.DataFrame):
                assert self.returns.columns.equals(
                    self.weights.columns  # type: ignore
                ), "returns and weights must have same columns"
                if rebalance_freq == "custom":
                    self.weights = weights
                else:
                    self.weights = (
                        weights.resample(constants.SAMPLING[rebalance_freq])
                        .last()
                        .loc[self.returns.index[0] : self.returns.index[-1]]
                    )
            elif isinstance(weights, (list, pd.Series)):
                assert len(weights) == len(self.assets), (
                    "weights must have same length as assets"
                )
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
        else:
            self.weights = None

        if weights is not None:
            self._calc_portfolio_returns()

        return None

    def _validate_rebalance_freq(self, rebalance_freq):
        assert rebalance_freq in constants.SAMPLING or rebalance_freq == "custom", (
            f"Invalid rebalance frequency! Choose from {list(constants.SAMPLING.keys()) + ['custom']}"
        )
        return None

    def update_rebalance_freq(self, rebalance_freq: str):
        assert self.weights is not None, "Weights are not set"
        self._validate_rebalance_freq(rebalance_freq)
        self.rebalance_freq = rebalance_freq
        self.set_weights(self.weights, rebalance_freq)
        return None

    def perf_summary(self, benchmark=None, yr=constants.YEAR_BY["day"]):
        if benchmark is None:
            benchmark = self.benchmark
        return backtest.perf_summary_table(self.portfolio_returns, bmk=benchmark, yr=yr)

    # TODO: Use scipy.optimize.minimize to optimize weights
    def mvo_weights(self, mu=None, sigma=None, rebalance_freq=None):
        """Calculate weights for a portfolio that maximizes the Sharpe ratio."""
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

        weights = sigma.quant.pinv().dot(mu)
        return weights.div(weights.sum())

    def kelly_weights(self, rebalance_freq=None, constraints=None):
        """Calculate optimal weights using the Kelly criterion by maximizing log returns.

        Args:
            rebalance_freq: Optional frequency for returns aggregation
            constraints: Optional dictionary of constraints for the optimization
                        (e.g., `bounds` on weights, `constraints` on sum of weights)
                        Default: weights sum to 1 and are between 0 and 1

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

        # Default constraints: weights sum to 1 and are non-negative
        if constraints is None:
            constraints = [
                {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # weights sum to 1
            ]
            bounds = [(0, 1) for _ in range(n_assets)]  # weights between 0 and 1
        else:
            bounds = constraints.get("bounds", [(0, 1) for _ in range(n_assets)])
            constraints = constraints.get(
                "constraints", [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
            )

        # Initial guess: equal weights
        initial_weights = np.array([1 / n_assets] * n_assets)

        # Optimize
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

    # TODO: Implement risk parity weights
    def risk_parity_weights(self):
        pass
