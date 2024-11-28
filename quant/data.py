"""APIs to access data"""

import datetime as dt
from functools import lru_cache
from typing import Union

import numpy as np
import pandas as pd
import requests
from mftool import Mftool

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
        fund_house = data.iloc[:, 0].name[1]
        fund_name = data.iloc[:, 0].name[3]
        col = fund_house + "|" + fund_name

        nav_df = (
            data.set_axis([col, "null1", "null2", "date"], axis=1)
            .set_index("date")
            .drop(["null1", "null2"], axis=1)
        )
        nav_df.index = pd.to_datetime(nav_df.index, dayfirst=True, format="mixed")
    except:
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
    if len(df) != 0:
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


def mftool(scid: Union[str, float, int], edate=dt.datetime.now()):
    mf = Mftool()
    data = mf.get_scheme_historical_nav(scid)
    try:
        col = data["fund_house"] + "|" + data["scheme_name"]
    except:
        print(f"Could not find data for {scid}")
    nav_df = pd.DataFrame(data["data"]).set_index("date").rename({"nav": col}, axis=1)
    nav_df.index = pd.to_datetime(nav_df.index, dayfirst=True)
    return nav_df.sort_index().astype("float").loc[:edate]


def fred(id: str):
    """Get timeseries from FRED"""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
    df = pd.read_csv(url).rename({"DATE": "date"}, axis=1).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df


@lru_cache(None)
def get_rfr(freq: str = "D"):
    assert freq in ["D", "W", "M", "Y"], "Input valid frequency!"
    raw_data = (
        pd.DataFrame().quant.fred("IRSTCI01INM156N").div(100).reset_index().to_numpy()
    )
    df = pd.DataFrame(
        np.concatenate([raw_data, [[dt.datetime.now(), np.nan]]]),
        columns=["date", "Cash"],
    ).set_index("date")
    df.index = pd.to_datetime(df.index.strftime("%Y-%m-%d"))
    df = df.resample("B").ffill().div(const.YEAR_BY["day"])
    sampling = {"D": "B", "W": "W-Wed", "M": "M", "Y": "Y"}
    return df.resample(sampling[freq]).sum().iloc[:-1]
