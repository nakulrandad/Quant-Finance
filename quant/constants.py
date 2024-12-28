"""All constants are defined here"""

YEAR_BY = {
    "month": 12,
    "week": 52,
    "day": 252,
    "cash_day": 360,
}

RISK_FREE_RATE_FRED = {
    "INR": "IRSTCI01INM156N",  # Monthly Call Money/Interbank Rate
    "USD": "DFF",  # Daily Effective Federal Funds Rate
}

SAMPLING = {"D": "B", "W": "W-Wed", "M": "ME", "Y": "Y"}
