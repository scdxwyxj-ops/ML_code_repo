"""Dataset loading helpers for experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from .config import CONFIG_PATH

STOCK_DIR = CONFIG_PATH.parent / "datasets" / "stock_market"
CANDIDATE_DIRS = [
    STOCK_DIR / "Data" / "Stocks",
    STOCK_DIR / "Data" / "ETFs",
    STOCK_DIR / "Stocks",
    STOCK_DIR / "ETFs",
]


def load_stock_market_data(
    tickers: Iterable[str],
    *,
    date_column: str = "Date",
    ticker_column: str = "Ticker",
    parse_dates: bool = True,
    limit_per_ticker: int | None = None,
) -> pd.DataFrame:
    """Load and concatenate stock data for the given ``tickers``."""

    frames = []
    for ticker in tickers:
        file_path = _locate_ticker_file(ticker)
        if not file_path:
            raise FileNotFoundError(f"Ticker file not found for {ticker}")

        read_kwargs: Mapping[str, object] = {
            "dtype": {
                "Open": "float64",
                "High": "float64",
                "Low": "float64",
                "Close": "float64",
                "Volume": "float64",
            },
        }
        if parse_dates:
            read_kwargs["parse_dates"] = [date_column]

        frame = pd.read_csv(file_path, **read_kwargs)
        if parse_dates and frame[date_column].dtype == "object":
            frame[date_column] = pd.to_datetime(frame[date_column])
        frame[ticker_column] = ticker
        if limit_per_ticker:
            frame = frame.head(limit_per_ticker)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(by=[ticker_column, date_column]).reset_index(drop=True)
    return combined


def _locate_ticker_file(ticker: str) -> Path | None:
    for directory in CANDIDATE_DIRS:
        candidate = directory / f"{ticker}.txt"
        if candidate.exists():
            return candidate
    return None


__all__ = ["load_stock_market_data"]
