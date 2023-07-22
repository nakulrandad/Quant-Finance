"""A flavour of pandas
"""
import numpy as np
import pandas as pd

import quant.constants as const

@pd.api.extensions.register_dataframe_accessor("quant")
class QuantDataFrameAccessor():
    """Quant DataFrame Accessor
    """
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        
    def return_mean(self, yr=const.YEAR_BY["day"]):
        """Returns annualized returns for a timeseries
        """
        mean = self._obj.mean() * yr
        return mean
    
    def return_vol(self, yr=const.YEAR_BY["day"]):
        """Returns annualized volatility for a timeseries
        """
        vol = self._obj.std() * np.sqrt(yr)
        return vol
    
    def sharpe(self, yr=const.YEAR_BY["day"], rfr=const.RISK_FREE_RATE["India"]):
        """Returns sharpe ratio of the timeseries
        
        rfr: annualized risk free rate
        """
        x = self._obj
        ex_ret = x.quant.return_mean(yr) - rfr
        sharpe = ex_ret/ x.quant.return_vol(yr)
        return sharpe
        
    
