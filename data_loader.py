"""Module 1: Data ingestion pipeline.

Downloads adjusted prices from Yahoo Finance, converts them to month-end
returns, and aligns a monthly risk-free rate from FRED (3-Month T-Bill, TB3MS).
"""

from __future__ import annotations

import io
import logging
import urllib.request
import warnings
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

# yfinance logs failed downloads at ERROR level; we raise our own clearer errors.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

DEFAULT_ANNUAL_RF = 0.04          # fallback when FRED is unreachable
MIN_MONTHS = 12                   # minimum history for meaningful statistics
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}"
HTTP_TIMEOUT = 15


class DataLoadError(Exception):
    """Raised for invalid tickers, insufficient history, or download failures."""


@dataclass
class MarketData:
    returns: pd.DataFrame             # columns: Date, Fund_Return, Benchmark_Return, Risk_Free_Rate
    fund_ticker: str
    benchmark_ticker: str
    requested_years: int
    fund_name: str
    benchmark_name: str
    fund_type: str
    rf_source: str                    # "FRED TB3MS" or fallback description
    latest_rf_annual: float
    warnings: list[str] = field(default_factory=list)

    @property
    def start(self) -> pd.Timestamp:
        return self.returns["Date"].iloc[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.returns["Date"].iloc[-1]


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def _last_complete_month_end(today: date | None = None) -> pd.Timestamp:
    """Month-end of the last fully completed month (partial months distort stats)."""
    today = pd.Timestamp(today or date.today())
    return (today + pd.offsets.MonthEnd(0)) if today.is_month_end else (today - pd.offsets.MonthEnd(1))


def download_adjusted_prices(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Daily Adjusted Close prices, one column per ticker."""
    try:
        raw = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False,        # keep the explicit "Adj Close" column
            progress=False,
            threads=True,
        )
    except Exception as exc:  # network errors, Yahoo API changes, rate limits
        raise DataLoadError(f"Price download failed: {exc}") from exc

    if raw is None or raw.empty:
        raise DataLoadError(f"No price data returned for {', '.join(tickers)}. Check the tickers and your connection.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" not in raw.columns.get_level_values(0):
            raise DataLoadError("Yahoo Finance response did not include Adjusted Close prices.")
        prices = raw["Adj Close"]
    else:  # single ticker, flat columns
        prices = raw[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})

    prices = prices.reindex(columns=tickers)
    missing = [t for t in tickers if prices[t].dropna().empty]
    if missing:
        raise DataLoadError(f"Invalid or unavailable ticker(s): {', '.join(missing)}")
    return prices


def to_monthly_returns(daily_prices: pd.DataFrame) -> pd.DataFrame:
    month_end_prices = daily_prices.resample("ME").last()
    return month_end_prices.pct_change(fill_method=None).iloc[1:]


def _security_info(ticker: str) -> tuple[str, str]:
    """(display name, vehicle type such as MUTUALFUND / ETF / EQUITY); best effort."""
    try:
        info = yf.Ticker(ticker).info or {}
        name = info.get("longName") or info.get("shortName") or ticker
        return name, info.get("quoteType") or "UNKNOWN"
    except Exception:
        return ticker, "UNKNOWN"


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------

def fetch_fred_series(series_id: str, start: pd.Timestamp) -> pd.Series:
    """Fetch a FRED series via pandas_datareader, falling back to FRED's CSV endpoint."""
    try:
        from pandas_datareader import data as pdr

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = pdr.DataReader(series_id, "fred", start)
        series = frame[series_id]
    except Exception:
        url = FRED_CSV_URL.format(series=series_id, start=start.strftime("%Y-%m-%d"))
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as resp:
            frame = pd.read_csv(io.BytesIO(resp.read()), index_col=0, parse_dates=True)
        series = frame.iloc[:, 0]

    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        raise ValueError(f"FRED series {series_id} returned no data")
    return series


def monthly_risk_free(index: pd.DatetimeIndex) -> tuple[pd.Series, str, float]:
    """Monthly risk-free rate aligned to `index`, plus source label and latest annual rate."""
    start = index[0] - pd.DateOffset(months=2)
    try:
        annual_pct = fetch_fred_series("TB3MS", start)
        by_month = annual_pct.copy()
        by_month.index = by_month.index.to_period("M")
        # FRED publishes with a lag, so carry the last known rate forward.
        aligned_pct = by_month.reindex(index.to_period("M")).ffill().bfill()
        aligned_pct.index = index
        source = "FRED TB3MS (3-Month T-Bill)"
        latest = float(annual_pct.iloc[-1]) / 100
    except Exception:
        aligned_pct = pd.Series(DEFAULT_ANNUAL_RF * 100, index=index)
        source = f"Fallback {DEFAULT_ANNUAL_RF:.1%} annualized (FRED unavailable)"
        latest = DEFAULT_ANNUAL_RF

    monthly = (1 + aligned_pct / 100) ** (1 / 12) - 1   # compounding-consistent conversion
    return monthly, source, latest


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def load_market_data(fund_ticker: str, benchmark_ticker: str, time_period_years: int) -> MarketData:
    fund_ticker = fund_ticker.strip().upper()
    benchmark_ticker = benchmark_ticker.strip().upper()
    if not fund_ticker or not benchmark_ticker:
        raise DataLoadError("Both a fund ticker and a benchmark ticker are required.")
    if fund_ticker == benchmark_ticker:
        raise DataLoadError("Fund and benchmark tickers must be different.")
    if time_period_years <= 0:
        raise DataLoadError("Time period must be a positive number of years.")

    end = _last_complete_month_end()
    start_month_end = end - pd.DateOffset(years=time_period_years)
    # Pull a few extra days so the starting month-end price is available.
    download_start = start_month_end - pd.Timedelta(days=10)

    prices = download_adjusted_prices([fund_ticker, benchmark_ticker], download_start, end)
    monthly = to_monthly_returns(prices)
    months = monthly.index.to_period("M")
    monthly = monthly[(months > start_month_end.to_period("M")) & (months <= end.to_period("M"))]

    notes: list[str] = []
    before = len(monthly)
    monthly = monthly.dropna()
    if len(monthly) < before:
        notes.append(
            f"Only {len(monthly)} of {before} requested months have data for both securities "
            f"(likely a shorter track record); analysis uses the common period."
        )
    if len(monthly) < MIN_MONTHS:
        raise DataLoadError(
            f"Only {len(monthly)} months of overlapping history; at least {MIN_MONTHS} are required."
        )

    rf, rf_source, latest_rf = monthly_risk_free(monthly.index)
    if rf_source.startswith("Fallback"):
        notes.append("Risk-free rate uses a 4.0% fallback because FRED could not be reached.")

    df = pd.DataFrame(
        {
            "Date": monthly.index,
            "Fund_Return": monthly[fund_ticker].to_numpy(dtype=float),
            "Benchmark_Return": monthly[benchmark_ticker].to_numpy(dtype=float),
            "Risk_Free_Rate": rf.to_numpy(dtype=float),
        }
    ).reset_index(drop=True)

    if not np.isfinite(df[["Fund_Return", "Benchmark_Return"]].to_numpy()).all():
        raise DataLoadError("Return series contains invalid values after cleaning.")

    fund_name, fund_type = _security_info(fund_ticker)
    return MarketData(
        returns=df,
        fund_ticker=fund_ticker,
        benchmark_ticker=benchmark_ticker,
        requested_years=time_period_years,
        fund_name=fund_name,
        benchmark_name=_security_info(benchmark_ticker)[0],
        fund_type=fund_type,
        rf_source=rf_source,
        latest_rf_annual=latest_rf,
        warnings=notes,
    )


def fetch_macro_snapshot() -> dict[str, str]:
    """Best-effort snapshot of the current macro backdrop from FRED and Yahoo.

    Every item is optional: anything that fails to download is simply omitted,
    so the AI commentary only ever sees real, dated figures.
    """
    snapshot: dict[str, str] = {}
    start = pd.Timestamp.today() - pd.DateOffset(years=2)

    def latest(series_id: str) -> tuple[float, pd.Timestamp] | None:
        try:
            s = fetch_fred_series(series_id, start)
            return float(s.iloc[-1]), s.index[-1]
        except Exception:
            return None

    for series_id, label in [
        ("FEDFUNDS", "Effective Fed Funds rate"),
        ("TB3MS", "3-Month T-Bill yield"),
        ("GS10", "10-Year Treasury yield"),
        ("UNRATE", "Unemployment rate"),
    ]:
        point = latest(series_id)
        if point:
            snapshot[label] = f"{point[0]:.2f}% (as of {point[1]:%b %Y})"

    try:
        cpi = fetch_fred_series("CPIAUCSL", start)
        yoy = cpi.iloc[-1] / cpi.iloc[-13] - 1
        snapshot["CPI inflation (YoY)"] = f"{yoy:.1%} (as of {cpi.index[-1]:%b %Y})"
    except Exception:
        pass

    gs10, tb3 = latest("GS10"), latest("TB3MS")
    if gs10 and tb3:
        snapshot["10Y minus 3M yield curve"] = f"{gs10[0] - tb3[0]:+.2f} pp"

    # Growth vs. value leadership: trailing 12-month total return, IWF vs IWD.
    try:
        end = _last_complete_month_end()
        px = download_adjusted_prices(["IWF", "IWD"], end - pd.DateOffset(months=12, days=10), end)
        m = px.resample("ME").last().dropna()
        trailing = m.iloc[-1] / m.iloc[-13] - 1
        snapshot["Russell 1000 Growth (IWF) trailing 12M"] = f"{trailing['IWF']:.1%}"
        snapshot["Russell 1000 Value (IWD) trailing 12M"] = f"{trailing['IWD']:.1%}"
    except Exception:
        pass

    return snapshot
