"""A flavour of pandas
"""
import datetime as dt

import numpy as np
import pandas as pd

import quant.data as data
import quant.constants as const


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

    def a2l(self):
        """Arithmatic to logarithmic returns"""
        x = self._obj
        log_r = np.log(x.add(1))
        return log_r

    def l2a(self):
        """Logarithmic to arithmatic returns"""
        x = self._obj
        return np.exp(x) - 1

    def t2e(self):
        """Convert total returns to excess return"""
        x = self._obj
        cash = pd.Series().cash()
        return x.subtract(cash, axis=0)

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

    @staticmethod
    def cash(tenor: str = "B"):
        """Get risk-free rate"""
        return data.get_rfr(tenor)

    @staticmethod
    def fred(id: str):
        """Get timeseries from FRED"""
        return data.fred(id)

    @staticmethod
    def mf(mfID, scID, edate=dt.datetime.now()):
        """Get historical nav of a mutual fund"""
        return data.amfi(mfID, scID, edate)


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

    def t2e(self):
        """Convert total returns to excess return"""
        x = self._obj
        cash = pd.Series().cash()
        return x.subtract(cash, axis=0)

    def max_drawdown(self):
        """Returns max drawdown for a return timeseries

        Returns
        -------
        list
            [drawdown_start, drawdown_end, max_drawdown]
        """
        ret = self._obj
        level = ret.dropna().add(1).cumprod()
        peak = [np.nan, -np.inf]
        mdd = [np.nan, np.nan, 0]
        for idx in level.index:
            if level.loc[idx] > peak[1]:
                peak[0] = idx
                peak[1] = level.loc[idx]
            dd = level.loc[idx] / peak[1] - 1
            if dd < mdd[2]:
                mdd[0] = peak[0]
                mdd[1] = idx
                mdd[2] = dd
        return mdd

    @staticmethod
    def fred(id: str):
        return pd.DataFrame().quant.fred(id).squeeze()

    @staticmethod
    def mf(mfID, scID, edate=dt.datetime.now()):
        return pd.DataFrame().quant.mf(mfID, scID, edate).squeeze()

    @staticmethod
    def cash(tenor: str = "B"):
        """Get risk-free rate"""
        return pd.DataFrame().quant.cash(tenor).squeeze()
