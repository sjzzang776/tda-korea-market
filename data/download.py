"""Download the raw price data used by the notebooks (not committed to the repository).

    pip install finance-datareader yfinance
    python data/download.py

Writes
  data/kr_indices_raw.csv        KOSPI, KOSPI200, KOSDAQ closes (FinanceDataReader / KRX), 1996-01 ->
  data/us_indices_raw.csv        S&P 500, DJIA, NASDAQ, Russell 2000 closes (FinanceDataReader / Yahoo), 1987-01 ->
  data/kr_sector_stocks_raw.csv  adjusted closes of the sector members in data/kr_sectors.json (yfinance), 2000-01 ->
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
END = "2026-10-20"


def indices():
    import FinanceDataReader as fdr
    kr = {n: fdr.DataReader(t, "1996-01-01", END)["Close"] for t, n in [("KS11", "KOSPI"), ("KS200", "KOSPI200"), ("KQ11", "KOSDAQ")]}
    pd.DataFrame(kr).to_csv(HERE / "kr_indices_raw.csv")
    us = {n: fdr.DataReader(t, "1987-01-01", END)["Close"] for t, n in [("^GSPC", "SP500"), ("^DJI", "DJIA"), ("^IXIC", "NASDAQ"), ("^RUT", "Russell2000")]}
    pd.DataFrame(us).to_csv(HERE / "us_indices_raw.csv")
    print("indices saved")


def sectors():
    import yfinance as yf
    sec = json.load(open(HERE / "kr_sectors.json", encoding="utf-8"))
    tickers = [t for m in sec.values() for t in m]
    px = yf.download(tickers, start="2000-01-01", end=END, progress=False, auto_adjust=True)["Close"]
    px.dropna(how="all").to_csv(HERE / "kr_sector_stocks_raw.csv")
    print("sector stocks saved", px.shape)


if __name__ == "__main__":
    indices()
    if (HERE / "kr_sectors.json").exists():
        sectors()
