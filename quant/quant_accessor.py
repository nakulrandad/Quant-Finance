"""A flavour of pandas
"""
import datetime as dt

import numpy as np
import pandas as pd

import quant.api as api
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

    @classmethod
    def fred(cls, id: str):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
        df = (
            pd.read_csv(url).rename({"DATE": "date"}, axis=1).set_index("date")
        )
        df.index = pd.to_datetime(df.index)
        return df

    @classmethod
    def amfi(cls, mfID, scID, edate=dt.datetime.now()):
        sdate = edate - dt.timedelta(days=5 * 360)
        df = api.amfi_api(mfID, scID, sdate, edate)
        col = df.columns[0]
        if len(df) != 0:
            df2 = pd.DataFrame().quant.amfi(mfID, scID, sdate)
            df = (
                pd.concat(
                    [
                        df2.set_axis(["nav"], axis=1),
                        df.set_axis(["nav"], axis=1),
                    ]
                )
                .sort_index()
                .set_axis([col], axis=1)
                .drop_duplicates(keep="last")
            )
        return df
