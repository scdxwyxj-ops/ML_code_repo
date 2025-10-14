"""Feature engineering helpers driven by configuration metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .config import CONFIG_PATH

PROJECT_ROOT = CONFIG_PATH.parent.parent
LOG_CLIP_LOWER = -0.999999
SHIFT_PERIODS = 1


def _safe_log1p(series: pd.Series) -> pd.Series:
    """Compute log1p on a series while guarding against invalid inputs."""

    sanitized = series.replace([np.inf, -np.inf], np.nan)
    clipped = sanitized.clip(lower=LOG_CLIP_LOWER)
    return np.log1p(clipped)


def _apply_shift_inplace(
    df: pd.DataFrame,
    columns: Iterable[str],
    groupby: Iterable[str] | None,
    periods: int = SHIFT_PERIODS,
) -> None:
    columns = [column for column in columns if column in df.columns]
    if not columns:
        return

    if groupby:
        df[columns] = df.groupby(list(groupby))[columns].shift(periods)
    else:
        df[columns] = df[columns].shift(periods)


def apply_feature_steps(df: pd.DataFrame, steps: Iterable[Mapping[str, object]] | None) -> pd.DataFrame:
    """Apply a sequence of feature engineering steps."""

    if not steps:
        return df

    working = df.copy()
    for step in steps:
        step_type = (step.get("type") or "").lower()
        if step_type == "technical_indicators":
            working = _add_technical_indicators(working, step)
        elif step_type == "sentiment_proxy":
            working = _add_sentiment_proxy(working, step)
        elif step_type == "macro_proxy":
            working = _add_macro_proxy(working, step)
        elif step_type == "external_join":
            working = _join_external_features(working, step)
        else:
            raise ValueError(f"Unsupported feature engineering step: {step_type}")

    return working.replace([np.inf, -np.inf], np.nan)


def _add_technical_indicators(df: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    price_column = config.get("price_column") or "Close"
    volume_column = config.get("volume_column") or "Volume"
    windows = config.get("windows") or [5, 10, 20]
    groupby = config.get("groupby")

    working = df.copy()

    if groupby:
        grouped_price = working.groupby(list(groupby))[price_column]
        returns = grouped_price.pct_change()
    else:
        grouped_price = None
        returns = working[price_column].pct_change()

    returns = returns.replace([np.inf, -np.inf], np.nan)
    working[f"{price_column}_return_1"] = returns
    working[f"{price_column}_log_return_1"] = _safe_log1p(returns)

    for window in windows:
        suffix = f"{window}"
        if grouped_price is not None:
            working[f"{price_column}_sma_{suffix}"] = grouped_price.transform(
                lambda series: series.rolling(window=window, min_periods=1).mean()
            )
            working[f"{price_column}_volatility_{suffix}"] = grouped_price.transform(
                lambda series: series.rolling(window=window, min_periods=1).std(ddof=0)
            )
            working[f"{price_column}_momentum_{suffix}"] = grouped_price.transform(
                lambda series: series.pct_change(periods=window)
            )
            working[f"{price_column}_rsi_{suffix}"] = grouped_price.transform(
                lambda series: _compute_rsi(series, window)
            )
        else:
            rolling_price = working[price_column].rolling(window=window, min_periods=1)
            working[f"{price_column}_sma_{suffix}"] = rolling_price.mean()
            working[f"{price_column}_volatility_{suffix}"] = rolling_price.std(ddof=0)
            working[f"{price_column}_momentum_{suffix}"] = working[price_column].pct_change(periods=window)
            working[f"{price_column}_rsi_{suffix}"] = _compute_rsi(working[price_column], window)

        if volume_column in working.columns:
            if groupby:
                grouped_volume = working.groupby(list(groupby))[volume_column]
                working[f"{volume_column}_sma_{suffix}"] = grouped_volume.transform(
                    lambda series: series.rolling(window=window, min_periods=1).mean()
                )
                working[f"{volume_column}_change_{suffix}"] = grouped_volume.transform(
                    lambda series: series.pct_change(periods=window)
                )
            else:
                rolling_volume = working[volume_column].rolling(window=window, min_periods=1)
                working[f"{volume_column}_sma_{suffix}"] = rolling_volume.mean()
                working[f"{volume_column}_change_{suffix}"] = working[volume_column].pct_change(periods=window)

    return working.replace([np.inf, -np.inf], np.nan)


def _add_sentiment_proxy(df: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    price_column = config.get("price_column") or "Close"
    groupby = config.get("groupby")
    window = int(config.get("window", 10))
    prefix = config.get("prefix") or "sentiment"

    working = df.copy()
    new_columns = [
        f"{prefix}_return_mean_{window}",
        f"{prefix}_return_std_{window}",
        f"{prefix}_ema_return_{window}",
    ]

    if groupby:
        groups = working.groupby(list(groupby))[price_column]
        working[new_columns[0]] = groups.transform(
            lambda series: series.pct_change().rolling(window=window, min_periods=1).mean()
        )
        working[new_columns[1]] = groups.transform(
            lambda series: series.pct_change().rolling(window=window, min_periods=1).std(ddof=0)
        )
        working[new_columns[2]] = groups.transform(
            lambda series: series.pct_change().ewm(span=window, adjust=False).mean()
        )
    else:
        pct_change = working[price_column].pct_change()
        working[new_columns[0]] = pct_change.rolling(window=window, min_periods=1).mean()
        working[new_columns[1]] = pct_change.rolling(window=window, min_periods=1).std(ddof=0)
        working[new_columns[2]] = pct_change.ewm(span=window, adjust=False).mean()

    working[new_columns] = working[new_columns].replace([np.inf, -np.inf], np.nan)
    _apply_shift_inplace(working, new_columns, groupby)
    return working


def _add_macro_proxy(df: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    date_column = config.get("date_column") or "Date"
    columns = config.get("columns") or ["Close", "Volume"]
    aggregations = config.get("aggregations") or ["mean", "std"]
    prefix = config.get("prefix") or "macro"
    groupby = config.get("groupby") or [date_column]

    if isinstance(groupby, str):
        groupby = [groupby]

    existing_columns = [column for column in columns if column in df.columns]
    if not existing_columns:
        return df

    agg_dict = {column: aggregations for column in existing_columns}
    aggregated = df.groupby(groupby).agg(agg_dict).reset_index()

    flattened_columns: list[str] = []
    for col in aggregated.columns:
        if isinstance(col, tuple):
            base, agg = col
            flattened_columns.append(f"{prefix}_{base}_{agg}" if agg else base)
        else:
            flattened_columns.append(col)
    aggregated.columns = flattened_columns

    merged = df.merge(aggregated, on=groupby, how="left")
    macro_columns = [column for column in merged.columns if column.startswith(f"{prefix}_")]
    shift_groupby = ["Ticker"] if "Ticker" in merged.columns else None
    _apply_shift_inplace(merged, macro_columns, shift_groupby)
    merged[macro_columns] = merged[macro_columns].replace([np.inf, -np.inf], np.nan)
    return merged


def _compute_rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0.0)


def _join_external_features(df: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    path = config.get("path")
    if not path:
        raise ValueError("external_join step requires a 'path'")

    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = (PROJECT_ROOT / path_obj).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"External feature file not found: {path_obj}")

    parse_dates = config.get("parse_dates")
    if parse_dates:
        external = pd.read_csv(path_obj, parse_dates=list(parse_dates))
    else:
        external = pd.read_csv(path_obj)

    on = config.get("on") or []
    how = config.get("how") or "left"

    merged = df.merge(external, on=on, how=how)

    fill_missing = config.get("fill_missing")
    if fill_missing is not None:
        numeric_cols = merged.select_dtypes(include=["number"]).columns
        merged[numeric_cols] = merged[numeric_cols].fillna(fill_missing)

    merged = merged.replace([np.inf, -np.inf], np.nan)
    return merged

def process_q3_features (df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        raise ValueError("Dataframe not found!")
    
    retained_cols = ["loan_status", "loan_amnt", "annual_inc", "annual_inc_joint", 
                     "fico_range_high", "fico_range_low", "dti_joint",
                     "dti", "revol_util", "purpose", "home_ownership", 
                     "emp_length", "term", "issue_d", "application_type", 
                     "int_rate", "total_acc", "open_acc", "delinq_2yrs", 
                     "acc_now_delinq", "acc_open_past_24mths", "verification_status", 
                     "pub_rec"]

    # Focus on relevant columns
    df = df[retained_cols].copy()

    # Handle Loan Issue Date for temporal stability analysis
    df = df.dropna(subset=['issue_d'])
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
    df['issue_year'] = df['issue_d'].dt.year
    df['issue_month'] = df['issue_d'].dt.month
    df = df.drop(columns=['issue_d'])
    
    # Preserve Fully Paid and Charged Off loan status types (non-ongoing loans)
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]

    # Handle Debt to Income Ratio (if joint DTI NaN, use DTI)
    df['dti_mod'] = df['dti_joint'].fillna(df['dti'])
    df = df.drop(columns=['dti_joint', 'dti']) 

    # Handle Annual Income (if joint AnnInc NaN, use AnnInc)
    df['annual_inc_mod'] = df['annual_inc_joint'].fillna(df['annual_inc'])
    df = df.drop(columns=['annual_inc_joint']) 

    # Handle other NaN types
    df = df.dropna() 

    # Evaluate Loan Income Ratio 
    median_income = df['annual_inc'].median()
    df['loan_income_ratio'] = df['loan_amnt'] / df['annual_inc'].replace(0, median_income) # Use median income if 0
    df = df.drop(columns=['loan_amnt', 'annual_inc']) 

    # Evaluate FICO mean from high and low FICO values
    df['fico_mean'] = 0.5*df['fico_range_high'] + 0.5*df['fico_range_low']
    df = df.drop(columns=['fico_range_high', 'fico_range_low'])

    df = df.reset_index(drop=True)

    return df



__all__ = ["apply_feature_steps", "process_q3_features"]
