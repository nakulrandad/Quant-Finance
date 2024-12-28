"""Portfolio class"""

from typing import Union

import numpy as np
import pandas as pd

from . import backtest, constants, utils


class Portfolio:
    def __init__(
        self,
        returns: Union[pd.DataFrame, list],  # as of period end
        weights: Union[pd.DataFrame, list, None] = None,  # as of period start
        rebalance_freq: str = "M",
        no_rebalance: bool = False,
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

        self.assets = self.returns.columns

        if weights is not None:
            if isinstance(weights, pd.DataFrame):
                assert self.returns.columns.equals(
                    self.weights.columns  # type: ignore
                ), "returns and weights must have same columns"
                if no_rebalance:
                    self.weights = weights
                else:
                    self.weights = (
                        weights.resample(constants.SAMPLING[rebalance_freq])
                        .last()
                        .loc[self.returns.index[0] : self.returns.index[-1]]
                    )
            elif isinstance(weights, list):
                assert len(weights) == len(
                    self.assets
                ), "weights must have same length as assets"
                self.weights = pd.DataFrame(
                    np.ones((len(self.returns), len(self.assets))),
                    index=self.returns.index,
                    columns=self.assets,
                ).mul(weights, axis=1)
                if not no_rebalance:
                    self.weights = (
                        self.weights.resample(constants.SAMPLING[rebalance_freq])
                        .last()
                        .loc[: self.returns.index[-1]]
                    )
        else:
            self.weights = None

        if weights is not None:
            self.calc_portfolio_returns()

        self.benchmark = benchmark

    def calc_effective_weights(self):
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

    def calc_portfolio_returns(self):
        self.calc_effective_weights()
        self.portfolio_returns = (
            (self.eff_weights * self.returns).sum(axis=1).to_frame("portfolio")
        )

    def perf_summary(self, bmk=None, yr=constants.YEAR_BY["day"]):
        if bmk is None:
            bmk = self.benchmark
        return backtest.perf_summary_table(self.portfolio_returns, bmk=bmk, yr=yr)
