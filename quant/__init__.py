"""Main quant package"""

from . import backtest, data, plots, quant_accessor, utils, version

__version__ = version.version
__author__ = "Nakul Randad"

__all__ = ["data", "backtest", "utils", "plots"]
