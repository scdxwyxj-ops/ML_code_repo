"""Dataset loading helpers and base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Mapping

import pandas as pd
import os
import math

from .config import CONFIG_PATH
from .feature_engineering import process_q3_features


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

class LendingClubDataset(BaseDataset):

    LENDING_CLUB_DIR = CONFIG_PATH.parent / "datasets" / "lending_club_loans"

    ACCEPTED_LOANS_PATH = LENDING_CLUB_DIR / "accepted_2007_to_2018q4.csv" /  "accepted_2007_to_2018q4.csv"
    
    def build_yearly_samples(self, chunk_size, samples_per_year, folder_path):

        if not self.LENDING_CLUB_DIR.exists():
            raise FileNotFoundError(f"Lending club dataset not found: {self.LENDING_CLUB_DIR}")
        elif not self.ACCEPTED_LOANS_PATH.exists():
            raise FileNotFoundError(f"Lending club accepted loans dataset not found: {self.ACCEPTED_LOANS_PATH}")

        yearly_samples = {}
        issue_year_col = 'issue_year'
        result_col = 'loan_status'
        samples_per_class = int(round(samples_per_year / 2))
        self.folder_path = folder_path

        os.makedirs(folder_path, exist_ok=True)

        for i, chunk in enumerate(pd.read_csv(self.ACCEPTED_LOANS_PATH, chunksize=chunk_size), start=1):

            print(f"Chunk {i} in progress!")

            chunk_processed = process_q3_features(chunk)

            for year, val in chunk_processed.groupby(issue_year_col):
                if year not in yearly_samples:
                    yearly_samples[year] = pd.DataFrame(columns=chunk_processed.columns)
                
                for credit_risk_class, cls_df in val.groupby(result_col):

                    num_cls_samples_now = (yearly_samples[year][result_col] == credit_risk_class).sum()

                    samples_to_go = samples_per_class - num_cls_samples_now

                    if samples_to_go <=0:
                        continue

                    samples = cls_df.sample(n=min(len(cls_df), samples_to_go), random_state=10)

                    yearly_samples[year] = pd.concat([yearly_samples[year], samples])
                
        for year, df in yearly_samples.items():
            file_name = f"lending_club_{year}.pkl"

            file_path = os.path.join(folder_path, file_name)

            if os.path.exists(file_path):
                os.remove(file_path)

            df.to_pickle(file_path)

            print(f"{file_name} saved! in {folder_path}!")
    
    def load(self, year: str) -> pd.DataFrame:
        file_name = f"lending_club_{year}.pkl"
        file_path = os.path.join(self.folder_path, file_name)

        df = pd.read_pickle(file_path)

        return df


# Backwards compatible helper functions -------------------------------------


def load_stock_market_data(**options: object) -> pd.DataFrame:
    return StockMarketDataset().load(options)


def load_credit_card_data(**options: object) -> pd.DataFrame:
    return CreditCardDataset().load(options)
