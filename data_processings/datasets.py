"""Dataset loading helpers and base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Mapping

import pandas as pd

from .config import CONFIG_PATH


class BaseDataset(ABC):
    """Abstract dataset loader interface."""

    @abstractmethod
    def load(self, options: Mapping[str, object]) -> pd.DataFrame:
        raise NotImplementedError


class StockMarketDataset(BaseDataset):
    STOCK_DIR = CONFIG_PATH.parent / "datasets" / "stock_market"
    CANDIDATE_DIRS = [
        STOCK_DIR / "Data" / "Stocks",
        STOCK_DIR / "Data" / "ETFs",
        STOCK_DIR / "Stocks",
        STOCK_DIR / "ETFs",
    ]

    def load(self, options: Mapping[str, object]) -> pd.DataFrame:
        tickers = options.get("tickers")
        if not tickers:
            raise ValueError("StockMarketDataset requires a 'tickers' list in options.")
        date_column = options.get("date_column", "Date")
        ticker_column = options.get("ticker_column", "Ticker")
        parse_dates = bool(options.get("parse_dates", True))
        limit_per_ticker = options.get("limit_per_ticker")

        frames = []
        for ticker in tickers:
            file_path = self._locate_ticker_file(ticker)
            if not file_path:
                raise FileNotFoundError(f"Ticker file not found for {ticker}")

            read_kwargs = {
                "dtype": {
                    "Open": "float64",
                    "High": "float64",
                    "Low": "float64",
                    "Close": "float64",
                    "Volume": "float64",
                }
            }
            if parse_dates:
                read_kwargs["parse_dates"] = [date_column]

            frame = pd.read_csv(file_path, **read_kwargs)
            if parse_dates and frame[date_column].dtype == "object":
                frame[date_column] = pd.to_datetime(frame[date_column])
            frame[ticker_column] = ticker
            if limit_per_ticker:
                frame = frame.head(int(limit_per_ticker))
            frames.append(frame)

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(by=[ticker_column, date_column]).reset_index(drop=True)
        return combined

    def _locate_ticker_file(self, ticker: str) -> Path | None:
        for directory in self.CANDIDATE_DIRS:
            candidate = directory / f"{ticker}.txt"
            if candidate.exists():
                return candidate
        return None


class CreditCardDataset(BaseDataset):
    DATA_PATH = CONFIG_PATH.parent / "datasets" / "credit_card_fraud" / "creditcard.csv"

    def load(self, options: Mapping[str, object]) -> pd.DataFrame:
        if not self.DATA_PATH.exists():
            raise FileNotFoundError(f"Credit card fraud dataset not found: {self.DATA_PATH}")

        parse_dates = bool(options.get("parse_dates", False))
        limit_rows = options.get("limit_rows")

        read_kwargs = {}
        if parse_dates:
            read_kwargs["parse_dates"] = ["Time"]

        df = pd.read_csv(self.DATA_PATH, **read_kwargs)
        if limit_rows:
            df = df.head(int(limit_rows))
        return df.reset_index(drop=True)


# Backwards compatible helper functions -------------------------------------


def load_stock_market_data(**options: object) -> pd.DataFrame:
    return StockMarketDataset().load(options)


def load_credit_card_data(**options: object) -> pd.DataFrame:
    return CreditCardDataset().load(options)
