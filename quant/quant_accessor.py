"""A flavour of pandas"""

import datetime as dt

import numpy as np
import pandas as pd

from . import api, utils
from . import constants as const


@pd.api.extensions.register_dataframe_accessor("quant")
class QuantDataFrameAccessor:
    """Quant DataFrame Accessor"""

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

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
        cov = pd.concat([x, bmk], axis=1).quant.align().cov()
        beta = (
            cov.loc[x.columns, bmk.columns]
            @ cov.loc[bmk.columns, bmk.columns].quant.pinv()
        )
        return beta

    def d2m(self):
        """Daily to monthly returns"""
        x = self._obj
        return x.add(1).cumprod().resample("M").last().pct_change()

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
        er = x.subtract(pd.Series().cash(freq), axis=0)
        return er[er.first_valid_index() :]

    def e2t(self, freq="D"):
        """Convert excess return to total return"""
        x = self._obj
        tr = x.add(pd.Series().cash(freq), axis=0)
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
        df = (
            pd.concat(
                [
                    pd.DataFrame([np.ones(len(x.columns))], columns=x.columns),
                    x.add(1).cumprod().dropna(),
                ]
            )
            .pct_change()
            .dropna()
        )
        return df

    def rename(self, cols):
        x = self._obj
        return x.set_axis(cols, axis=1)

    def first_valid_index(self):
        x = self._obj
        fvi = []
        for col in x.columns:
            fvi.append({"asset": col, "date": x[col].first_valid_index()})
        return pd.DataFrame(fvi).sort_values(by="date").set_index("asset").squeeze()

    @staticmethod
    def cash(freq: str = "D"):
        """Get risk-free rate"""
        return api.get_rfr(freq)

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
    def cash(freq: str = "D"):
        """Get risk-free rate"""
        return pd.DataFrame().quant.cash(freq).squeeze()
