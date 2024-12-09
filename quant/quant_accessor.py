"""A flavour of pandas"""

import datetime as dt

import numpy as np
import pandas as pd

from . import api, utils
from . import constants as const


class SharedStateManager:
    """A singleton-like manager to store shared state."""

    def __init__(self):
        self._currency = "INR"

    def set_currency(self, value):
        assert value in ["INR", "USD"], "Currency must be either 'INR' or 'USD'."
        self._currency = value
        print(f"Default currency set to {value}")

    def get_currency(self):
        return self._currency


shared_manager = SharedStateManager()


@pd.api.extensions.register_dataframe_accessor("quant")
class QuantDataFrameAccessor:
    """Quant DataFrame Accessor"""

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @staticmethod
    def set_currency(value):
        shared_manager.set_currency(value)

    @staticmethod
    def get_currency():
        return shared_manager.get_currency()

    def return_mean(self, yr=const.YEAR_BY["day"]):
        """Returns annualized returns for a timeseries"""
        mean = self._obj.mean() * yr
        return mean

    def return_vol(self, yr=const.YEAR_BY["day"]):
        """Returns annualized volatility for a timeseries"""
        vol = self._obj.std() * np.sqrt(yr)
        return vol

    def sharpe(self, yr=const.YEAR_BY["day"]):
        """Returns sharpe ratio of the timeseries.
        Assumes excess return is passed."""
        x = self._obj
        sharpe = x.quant.return_mean(yr) / x.quant.return_vol(yr)
        return sharpe

    def pinv(self):
        """Partial inverse of dataframe"""
        x = self._obj
        pinv = pd.DataFrame(np.linalg.pinv(x), columns=x.columns, index=x.index)
        return pinv

    def beta(self, bmk: pd.DataFrame):
        """Portfolio beta to benchmark"""
        x = self._obj
        n = len(x.columns)
        cov = pd.concat([x, bmk], axis=1).quant.align().cov()
        beta = cov.iloc[:n, n:] @ cov.iloc[n:, n:].quant.pinv()
        return beta

    def alpha(self, bmk: pd.DataFrame, yr=const.YEAR_BY["day"]):
        """Portfolio alpha to benchmark"""
        x = self._obj
        beta = x.quant.beta(bmk)
        alpha = beta.mul(bmk.mean()).sum(axis=1) * yr
        return alpha

    def tracking_error(self, bmk: pd.DataFrame, yr=const.YEAR_BY["day"]):
        """Tracking error of the portfolio"""
        x = self._obj
        beta = x.quant.beta(bmk)
        tracking_error = pd.Series()
        for asset, b in beta.iterrows():
            tracking_error[asset] = (x[asset] - b.mul(bmk).sum(axis=1)).std()
        return tracking_error * np.sqrt(yr)

    def information_ratio(self, bmk: pd.DataFrame, yr=const.YEAR_BY["day"]):
        """Information ratio of the portfolio"""
        x = self._obj
        alpha = x.quant.alpha(bmk, yr)
        te = x.quant.tracking_error(bmk, yr)
        ir = alpha.div(te)
        return ir

    def hit_ratio(self, bmk=None):
        """Hit ratio of the portfolio"""
        x = self._obj
        hit_ratio = pd.Series()
        if bmk is None:
            for col in x.columns:
                hit_ratio[col] = x[col].dropna().gt(0).mean()
            return hit_ratio
        else:
            assert isinstance(bmk, pd.DataFrame), "Benchmark must be a DataFrame"
            assert bmk.shape[1] == 1, "Benchmark must have only one column"
            for col in x.columns:
                merged = pd.merge(
                    left=x[[col]],
                    right=bmk,
                    left_index=True,
                    right_index=True,
                    how="inner",
                ).quant.align()
                hit_ratio[col] = (merged.iloc[:, 0] > merged.iloc[:, 1]).mean()
            return hit_ratio

    def d2m(self):
        """Daily to monthly returns"""
        x = self._obj
        return x.add(1).cumprod().resample("ME").last().pct_change()

    def m2y(self):
        """Monthly to yearly returns"""
        x = self._obj
        return x.add(1).cumprod().resample("Y").last().pct_change()

    def a2l(self):
        """Arithmatic to logarithmic returns"""
        x = self._obj
        log_r = np.log(x.add(1))
        return log_r

    def l2a(self):
        """Logarithmic to arithmatic returns"""
        x = self._obj
        return np.exp(x) - 1

    def t2e(self, freq="D"):
        """Convert total return to excess return"""
        x = self._obj
        er = x.subtract(pd.Series().quant.cash(freq), axis=0)
        return er[er.first_valid_index() :]

    def e2t(self, freq="D"):
        """Convert excess return to total return"""
        x = self._obj
        tr = x.add(pd.Series().quant.cash(freq), axis=0)
        return tr[tr.first_valid_index() :]

    def to_returns(self):
        """Convert prices to returns"""
        x = self._obj
        x = utils.prepare_prices(x, fill_method="ffill")
        for col in x.columns:
            x[col] = x[col].pct_change()
        x = x.dropna(how="all")
        return x

    def to_prices(self):
        """Convert returns to prices"""
        x = self._obj
        x = utils.prepare_returns(x)
        for col in x.columns:
            x[col] = x[col].add(1).cumprod()
        return x

    def rebase(self, base=100):
        """Rebase price timeseries"""
        x = self._obj
        for col in x.columns:
            x[col] = x[col].div(x[col].iloc[0]).mul(base)
        return x

    def max_drawdown(self):
        """Returns max drawdown for return timeseries of each asset

        Returns
        -------
        dict
            {asset: [drawdown_start, drawdown_end, max_drawdown]}
        """
        ret = self._obj
        mdd = {}
        for col in ret.columns:
            mdd[col] = ret[col].quant.max_drawdown()
        return mdd

    def align(self):
        """Align returns dataframe"""
        x = self._obj
        fvi = x.quant.first_valid_index().iloc[0]
        lvi = x.quant.last_valid_index().iloc[0]
        df = x.loc[fvi:lvi].add(1).cumprod().ffill().pct_change().dropna(how="all")
        return df

    def rename(self, cols):
        x = self._obj
        return x.set_axis(cols, axis=1)

    def first_valid_index(self):
        x = self._obj
        fvi = []
        for col in x.columns:
            fvi.append({"asset": col, "date": x[col].first_valid_index()})
        return (
            pd.DataFrame(fvi)
            .sort_values(by="date", ascending=False)
            .set_index("asset")
            .squeeze()
        )

    def last_valid_index(self):
        x = self._obj
        lvi = []
        for col in x.columns:
            lvi.append({"asset": col, "date": x[col].last_valid_index()})
        return pd.DataFrame(lvi).sort_values(by="date").set_index("asset").squeeze()

    @staticmethod
    def cash(freq: str = "D", curr=None):
        """Get risk-free rate"""
        if curr is None:
            curr = shared_manager.get_currency()
        return api.get_rfr(freq, curr)

    @staticmethod
    def fred(id: str):
        """Get timeseries from FRED"""
        return api.fred_api(id)

    @staticmethod
    def mutual_fund(scheme_ids, end=dt.datetime.now()):
        """Get historical nav of a mutual fund"""
        if isinstance(scheme_ids, (int, float, str)):
            scheme_ids = [scheme_ids]
        data = []
        for scid in scheme_ids:
            data.append(api.mf_api(scid).loc[:end])
        return pd.concat(data, axis=1)

    @staticmethod
    def ticker(tickers, end=dt.datetime.now()):
        """Get historical prices of a ticker"""
        if isinstance(tickers, str):
            tickers = [tickers]
        data = []
        for ticker in tickers:
            series = api.yf_api(ticker)
            series.index.name = "date"
            series.name = ticker
            data.append(series)
        return pd.concat(data, axis=1).dropna(how="all").loc[:end]


@pd.api.extensions.register_series_accessor("quant")
class QuantSeriesAccessor:
    """Quant Series Accessor"""

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @staticmethod
    def set_currency(value):
        shared_manager.set_currency(value)

    @staticmethod
    def get_currency():
        return shared_manager.get_currency()

    def return_mean(self, yr=const.YEAR_BY["day"]):
        """Returns annualized returns for a timeseries"""
        mean = self._obj.mean() * yr
        return mean

    def return_vol(self, yr=const.YEAR_BY["day"]):
        """Returns annualized volatility for a timeseries"""
        vol = self._obj.std() * np.sqrt(yr)
        return vol

    def sharpe(self, yr=const.YEAR_BY["day"]):
        """Returns sharpe ratio of the timeseries.
        Assumes excess return is passed."""
        x = self._obj
        sharpe = x.quant.return_mean(yr) / x.quant.return_vol(yr)
        return sharpe

    def a2l(self):
        """Arithmatic to logarithmic returns"""
        x = self._obj
        log_r = np.log(x.add(1))
        return log_r

    def l2a(self):
        """Logarithmic to arithmatic returns"""
        x = self._obj
        return np.exp(x) - 1

    def t2e(self, freq: str = "D"):
        """Convert total returns to excess return"""
        x = self._obj
        er = x.subtract(pd.Series().cash(freq), axis=0)
        return er[er.first_valid_index() :]

    def e2t(self, freq: str = "D"):
        """Convert excess return to total return"""
        x = self._obj
        tr = x.add(pd.Series().cash(freq), axis=0)
        return tr[tr.first_valid_index() :]

    def max_drawdown(self):
        """Returns max drawdown for a return timeseries

        Returns
        -------
        list
            [drawdown_start, drawdown_end, max_drawdown]
        """
        ret = self._obj
        price = ret.dropna().add(1).cumprod()
        peak = [np.nan, -np.inf]
        mdd = [np.nan, np.nan, 0]
        for idx in price.index:
            if price.loc[idx] > peak[1]:
                peak[0] = idx
                peak[1] = price.loc[idx]
            dd = price.loc[idx] / peak[1] - 1
            if dd < mdd[2]:
                mdd[0] = peak[0]
                mdd[1] = idx
                mdd[2] = dd
        return mdd

    @staticmethod
    def fred(id: str):
        return pd.DataFrame().quant.fred(id).squeeze()

    @staticmethod
    def cash(freq: str = "D", curr=None):
        """Get risk-free rate"""
        if curr is None:
            curr = shared_manager.get_currency()
        return pd.DataFrame().quant.cash(freq, curr).squeeze()
