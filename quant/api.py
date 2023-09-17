"""APIs to access data
"""
import datetime as dt
import requests

import pandas as pd


def amfi_api(mfID, scID, fDate, tDate) -> pd.DataFrame:
    try:
        fDate = dt.datetime.strftime(fDate, "%d-%b-%Y")
        tDate = dt.datetime.strftime(tDate, "%d-%b-%Y")

        url = "https://www.amfiindia.com/modules/NavHistoryPeriod"
        params = {
            "mfID": str(mfID),
            "scID": str(scID),
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
        col = fund_house + " | " + fund_name

        nav_df = (
            data.set_axis([col, "null1", "null2", "date"], axis=1)
            .set_index("date")
            .drop(["null1", "null2"], axis=1)
        )
        nav_df.index = pd.to_datetime(
            nav_df.index, dayfirst=True, format="mixed"
        )
    except:
        nav_df = pd.DataFrame(columns=["date", "nav"]).set_index("date")
    return nav_df
