"""Main quant package"""

from . import api, backtest, plot, quant_accessor, utils, version

__version__ = version.version
__author__ = "Nakul Randad"

__all__ = ["api", "backtest", "utils", "plot"]
