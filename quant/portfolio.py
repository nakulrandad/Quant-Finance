"""Portfolio class"""

from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd

from . import backtest, constants, utils


class Portfolio:
    def __init__(
        self,
        returns: Union[pd.DataFrame, list],  # as of period end
        weights: Union[
            pd.DataFrame, pd.Series, list, None
        ] = None,  # as of period start
        rebalance_freq: str = "M",
        benchmark: Union[pd.DataFrame, None] = None,
    ):
        """A class to represent a group of asset returns. The assets can be assigned
        weights and rebalanced to create a portfolio for backtesting.
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
        return f"Portfolio(assets={self.assets}, start={self.returns.index[0]}, end={self.returns.index[-1]}, rebalance_freq={self.rebalance_freq})"

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

    def mvo_weights(self, mu=None, sigma=None, rebalance_freq=None):
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

    # TODO: Implement Kelly criterion
    def kelly_weights(self):
        pass

    # TODO: Implement risk parity weights
    def risk_parity_weights(self):
        pass
