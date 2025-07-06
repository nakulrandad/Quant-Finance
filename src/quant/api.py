"""APIs to access data"""

import datetime as dt
from functools import lru_cache
from typing import Union

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# from mftool import Mftool
from . import constants as const


def amfi_api(mfid, scid, fDate, tDate) -> pd.DataFrame:
    try:
        fDate = dt.datetime.strftime(fDate, "%d-%b-%Y")
        tDate = dt.datetime.strftime(tDate, "%d-%b-%Y")

        url = "https://www.amfiindia.com/modules/NavHistoryPeriod"
        params = {
            "mfID": str(mfid),
            "scID": str(scid),
            "fDate": fDate,
            "tDate": tDate,
        }
        headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        }
        r = requests.post(url=url, params=params, headers=headers)

        data = pd.read_html(r.text)[0]
        fund_house = data.iloc[:, 0].name[1]  # type: ignore
        fund_name = data.iloc[:, 0].name[3]  # type: ignore
        col = fund_house + "|" + fund_name

        nav_df = (
            data.set_axis([col, "null1", "null2", "date"], axis=1)
            .set_index("date")
            .drop(["null1", "null2"], axis=1)
        )
        nav_df.index = pd.to_datetime(nav_df.index, dayfirst=True, format="mixed")
    except Exception:
        nav_df = pd.DataFrame(columns=["date", "nav"]).set_index("date")
    return nav_df


def amfi(
    mfid: Union[str, float, int],
    scid: Union[str, float, int],
    edate=dt.datetime.now(),
):
    """Get historical nav of a mutual fund"""
    sdate = edate - dt.timedelta(days=5 * 360)
    df = amfi_api(mfid, scid, sdate, edate)
    col = df.columns[0]
    if not df.empty:
        df2 = amfi(mfid, scid, sdate)
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


# def mftool(scid: Union[str, float, int], edate=dt.datetime.now()):
#     mf = Mftool()
#     data = mf.get_scheme_historical_nav(scid)
#     try:
#         col = data["fund_house"] + "|" + data["scheme_name"]
#     except KeyError:
#         print(f"Could not find data for {scid}")
#     nav_df = pd.DataFrame(data["data"]).set_index("date").rename({"nav": col}, axis=1)
#     nav_df.index = pd.to_datetime(nav_df.index, dayfirst=True)
#     return nav_df.sort_index().astype("float").loc[:edate]


def mf_list(filter=None) -> pd.DataFrame:
    """Get list of mutual funds and sceheme ids
    filter : str or list
        Filter using case insensitive name
    """
    url = "https://api.mfapi.in/mf"
    data = requests.get(url).json()
    data = pd.DataFrame(data)
    if filter is not None:
        if isinstance(filter, str):
            filter = [filter]
        mask = data.schemeName.str.contains(filter[0], case=False)
        if len(filter) > 1:
            for f in filter[1:]:
                mask &= data.schemeName.str.contains(f, case=False)
        data = data[mask]
    return data


def mf_api(scid: Union[str, float, int]) -> pd.DataFrame:
    """Get historical nav of a mutual fund from mfapi"""
    if isinstance(scid, (int, float)):
        scid = str(int(scid))
    url = f"https://api.mfapi.in/mf/{scid}"

    data = requests.get(url).json()
    if not data["data"]:
        raise ValueError(f"Could not find data for {scid}")

    df = pd.DataFrame(data["data"]).set_index("date")
    df.index = pd.to_datetime(df.index, dayfirst=True)
    df = df.sort_index().rename(columns={"nav": data["meta"]["scheme_name"]})
    return df


def fred_api(id: str) -> pd.DataFrame:
    """Get timeseries from FRED"""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
    df = (
        pd.read_csv(url)
        .rename({"DATE": "date", "observation_date": "date"}, axis=1)
        .set_index("date")
    )
    df.index = pd.to_datetime(df.index)
    return df


@lru_cache(None)
def get_rfr(freq: str = "D", curr: str = "INR") -> pd.DataFrame:
    assert freq in ["D", "W", "M", "Y"], "Input valid frequency!"
    assert curr in ["INR", "USD"], "Input valid currency!"
    raw_data = (
        fred_api(const.RISK_FREE_RATE_FRED[curr]).div(100).reset_index().to_numpy()
    )
    df = pd.DataFrame(
        np.concatenate([raw_data, [[dt.datetime.now(), np.nan]]]),
        columns=["date", "Cash"],
    ).set_index("date")
    df.index = pd.to_datetime(df.index).normalize()
    df = df.resample("D").ffill().div(const.YEAR_BY["cash_day"])
    df = df.resample(const.SAMPLING[freq]).sum().iloc[:-1]
    return df.tz_localize(None)


def yf_api(ticker, period="max") -> pd.DataFrame:
    """Get ticker data from Yahoo Finance
    period : str
        Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    """
    params = {
        "tickers": ticker,
        "auto_adjust": True,
        "progress": False,
    }
    if isinstance(period, pd.DatetimeIndex):
        params["start"] = period[0]
    else:
        params["period"] = period
    df: pd.DataFrame = yf.download(**params)["Close"]  # type: ignore
    df = df.tz_localize(None)
    return df
