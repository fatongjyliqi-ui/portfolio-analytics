"""Market data loaders (Yahoo Finance prices, Fama-French factors).

Everything that needs an internet connection lives in this module, so the
rest of the package can be tested offline.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile

import pandas as pd

FF3_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_Factors_daily_CSV.zip"
)


def load_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Daily split- and dividend-adjusted close prices from Yahoo Finance.

    Only dates on which every ticker has a price are kept, so the columns
    are aligned on common trading days.
    """
    import yfinance as yf  # imported lazily: only needed for live data

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers
    return prices.dropna()


def load_ff3_daily() -> pd.DataFrame:
    """Daily Fama-French 3-factor data from Kenneth French's data library.

    Returns a DataFrame indexed by date with columns Mkt-RF, SMB, HML, RF,
    expressed as decimals (the source file is in percent).
    """
    with urllib.request.urlopen(FF3_DAILY_URL) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        with zf.open(zf.namelist()[0]) as csv_file:
            df = pd.read_csv(csv_file, skiprows=4)

    df = df.rename(columns={df.columns[0]: "date"})
    df = df[df["date"].astype(str).str.match(r"^\d{8}$", na=False)]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").astype(float) / 100
    df.columns = [c.strip() for c in df.columns]
    return df
