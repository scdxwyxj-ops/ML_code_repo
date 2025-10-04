"""Rolling window feature utilities."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import pandas as pd

SUPPORTED_ROLLING_AGGS = {
    "mean": lambda windowed: windowed.mean(),
    "std": lambda windowed: windowed.std(ddof=0),
    "sum": lambda windowed: windowed.sum(),
    "min": lambda windowed: windowed.min(),
    "max": lambda windowed: windowed.max(),
}


def apply_rolling_features(
    df: pd.DataFrame,
    configs: Iterable[Mapping[str, object]] | None,
) -> pd.DataFrame:
    """Derive rolling features as described by ``configs``."""

    if not configs:
        return df.copy()

    working = df.copy()
    for spec in configs:
        columns = [col for col in spec.get("columns", []) if col in working.columns]
        if not columns:
            continue

        window = int(spec.get("window", 5))
        min_periods = int(spec.get("min_periods", 1))
        aggs = [agg.lower() for agg in spec.get("aggregations", ["mean"])]
        prefix = spec.get("prefix") or "roll"
        groupby = spec.get("groupby")

        if groupby:
            grouped = working.groupby(list(groupby), group_keys=False)
            for column in columns:
                working = _apply_grouped_rolling(
                    working,
                    grouped[column],
                    column,
                    aggs,
                    window,
                    min_periods,
                    prefix,
                )
        else:
            for column in columns:
                series = working[column].rolling(window=window, min_periods=min_periods)
                working = _apply_rolling_series(working, series, column, aggs, window, prefix)

    return working


def _apply_grouped_rolling(
    df: pd.DataFrame,
    grouped_series: pd.core.groupby.generic.SeriesGroupBy,
    column: str,
    aggs: Sequence[str],
    window: int,
    min_periods: int,
    prefix: str,
) -> pd.DataFrame:
    for agg in aggs:
        func = SUPPORTED_ROLLING_AGGS.get(agg)
        if func is None:
            raise ValueError(f"Unsupported rolling aggregation: {agg}")

        new_column = f"{prefix}_{column}_{agg}_{window}"
        df[new_column] = grouped_series.transform(
            lambda series: func(series.rolling(window=window, min_periods=min_periods))
        )
    return df


def _apply_rolling_series(
    df: pd.DataFrame,
    rolling_obj: pd.core.window.rolling.Rolling,
    column: str,
    aggs: Sequence[str],
    window: int,
    prefix: str,
) -> pd.DataFrame:
    for agg in aggs:
        func = SUPPORTED_ROLLING_AGGS.get(agg)
        if func is None:
            raise ValueError(f"Unsupported rolling aggregation: {agg}")
        new_column = f"{prefix}_{column}_{agg}_{window}"
        df[new_column] = func(rolling_obj)
    return df


__all__ = ["apply_rolling_features", "SUPPORTED_ROLLING_AGGS"]
