"""Utilities for handling missing values across the assignment datasets."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence
from typing import Literal

import pandas as pd

from .config import CONFIG_PATH, load_config

NumericStrategy = Literal["mean", "median", "zero", "constant", "ffill", "bfill", "interpolate"]
CategoricalStrategy = Literal["mode", "constant", "ffill", "bfill"]
DatetimeStrategy = Literal["ffill", "bfill", "interpolate", "constant"]


@dataclass(frozen=True)
class MissingValueConfig:
    """Configuration bundle describing how to treat missing values."""

    numeric_strategy: NumericStrategy = "median"
    numeric_fill_value: Optional[float] = None
    numeric_columns: Optional[Sequence[str]] = None

    categorical_strategy: CategoricalStrategy = "mode"
    categorical_fill_value: Optional[Any] = None
    categorical_columns: Optional[Sequence[str]] = None

    datetime_strategy: Optional[DatetimeStrategy] = "ffill"
    datetime_fill_value: Optional[Any] = None
    datetime_columns: Optional[Sequence[str]] = None

    constant_fill: Optional[Mapping[str, Any]] = None
    drop_columns_threshold: Optional[float] = None
    drop_rows_threshold: Optional[float] = None
    indicator: bool = False
    exclude_columns: Optional[Sequence[str]] = None


def _load_missing_value_presets_from_disk() -> Dict[str, MissingValueConfig]:
    config = load_config()
    raw_presets = config.get("missing_value_presets")
    if not isinstance(raw_presets, dict):
        raise ValueError(
            "'missing_value_presets' must be a JSON object mapping dataset names to configs."
        )

    presets: Dict[str, MissingValueConfig] = {}
    for dataset_name, overrides in raw_presets.items():
        if not isinstance(overrides, dict):
            raise ValueError(f"Preset for '{dataset_name}' must be a JSON object.")
        presets[dataset_name] = MissingValueConfig(**overrides)

    return presets


@lru_cache(maxsize=1)
def _get_missing_value_presets() -> Dict[str, MissingValueConfig]:
    return _load_missing_value_presets_from_disk()


DATASET_MISSING_VALUE_PRESETS = _get_missing_value_presets()


def handle_missing_values(
    df: pd.DataFrame,
    *,
    config: Optional[MissingValueConfig] = None,
    inplace: bool = False,
    **overrides: Any,
) -> pd.DataFrame:
    """Apply a configurable set of missing value rules to ``df``."""

    resolved = _resolve_config(config, overrides)
    working = df if inplace else df.copy()

    _validate_threshold(resolved.drop_columns_threshold, "drop_columns_threshold")
    _validate_threshold(resolved.drop_rows_threshold, "drop_rows_threshold")

    exclude = set(resolved.exclude_columns or [])

    if resolved.drop_rows_threshold is not None:
        missing_fraction = working.isna().mean(axis=1)
        working = working.loc[missing_fraction <= resolved.drop_rows_threshold].copy()

    if resolved.drop_columns_threshold is not None:
        col_missing = working.isna().mean()
        to_drop = [
            col
            for col, frac in col_missing.items()
            if frac > resolved.drop_columns_threshold and col not in exclude
        ]
        if to_drop:
            working = working.drop(columns=to_drop)

    if resolved.constant_fill:
        _apply_constant_fill(working, resolved.constant_fill, resolved.indicator)

    numeric_cols = _select_columns(
        working,
        resolved.numeric_columns,
        include_dtypes=("number",),
        exclude=exclude,
    )
    categorical_cols = _select_columns(
        working,
        resolved.categorical_columns,
        include_dtypes=("object", "category", "string", "bool"),
        exclude=exclude,
    )
    datetime_cols = _select_columns(
        working,
        resolved.datetime_columns,
        include_dtypes=("datetime", "datetimetz"),
        exclude=exclude,
    )

    _apply_numeric_strategy(
        working,
        numeric_cols,
        resolved.numeric_strategy,
        resolved.numeric_fill_value,
        resolved.indicator,
    )
    _apply_categorical_strategy(
        working,
        categorical_cols,
        resolved.categorical_strategy,
        resolved.categorical_fill_value,
        resolved.indicator,
    )
    _apply_datetime_strategy(
        working,
        datetime_cols,
        resolved.datetime_strategy,
        resolved.datetime_fill_value,
        resolved.indicator,
    )

    return working


def get_missing_value_config(
    dataset_name: str,
    *,
    overrides: Optional[Mapping[str, Any]] = None,
) -> MissingValueConfig:
    """Return a preset configuration for the given dataset name."""

    presets = _get_missing_value_presets()
    base = presets.get(dataset_name)
    if base is None:
        raise KeyError(f"Unknown dataset preset: {dataset_name}")

    if not overrides:
        return base

    return replace(base, **{key: value for key, value in overrides.items() if value is not None})


def _resolve_config(
    config: Optional[MissingValueConfig],
    overrides: Mapping[str, Any],
) -> MissingValueConfig:
    if config is None:
        config = MissingValueConfig()

    if overrides:
        filtered = {key: value for key, value in overrides.items() if value is not None}
        if filtered:
            config = replace(config, **filtered)

    return config


def _validate_threshold(value: Optional[float], name: str) -> None:
    if value is None:
        return
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1 (received {value!r}).")


def _apply_constant_fill(
    df: pd.DataFrame,
    fill_map: Mapping[str, Any],
    indicator: bool,
) -> None:
    for column, fill_value in fill_map.items():
        if column not in df.columns:
            continue
        mask = df[column].isna()
        if indicator:
            _set_indicator(df, column, mask)
        df[column] = df[column].fillna(fill_value)


def _apply_numeric_strategy(
    df: pd.DataFrame,
    columns: Sequence[str],
    strategy: Optional[NumericStrategy],
    fill_value: Optional[float],
    indicator: bool,
) -> None:
    if not columns or strategy is None:
        return

    mask = df[columns].isna()
    if indicator:
        for column in columns:
            _set_indicator(df, column, mask[column])

    if strategy == "mean":
        fills = df[columns].mean()
        df[columns] = df[columns].fillna(fills)
    elif strategy == "median":
        fills = df[columns].median()
        df[columns] = df[columns].fillna(fills)
    elif strategy == "zero":
        df[columns] = df[columns].fillna(0.0 if fill_value is None else fill_value)
    elif strategy == "constant":
        if fill_value is None:
            raise ValueError("numeric_fill_value is required when using the 'constant' strategy.")
        df[columns] = df[columns].fillna(fill_value)
    elif strategy == "ffill":
        df[columns] = df[columns].ffill()
    elif strategy == "bfill":
        df[columns] = df[columns].bfill()
    elif strategy == "interpolate":
        df[columns] = df[columns].interpolate(limit_direction="both")
    else:
        raise ValueError(f"Unsupported numeric strategy: {strategy}")


def _apply_categorical_strategy(
    df: pd.DataFrame,
    columns: Sequence[str],
    strategy: Optional[CategoricalStrategy],
    fill_value: Optional[Any],
    indicator: bool,
) -> None:
    if not columns or strategy is None:
        return

    mask = df[columns].isna()
    if indicator:
        for column in columns:
            _set_indicator(df, column, mask[column])

    if strategy == "mode":
        fills = {}
        for column in columns:
            mode_series = df[column].mode(dropna=True)
            if not mode_series.empty:
                fills[column] = mode_series.iloc[0]
        if fills:
            df[columns] = df[columns].fillna(pd.Series(fills))
    elif strategy == "constant":
        if fill_value is None:
            raise ValueError(
                "categorical_fill_value is required when using the 'constant' strategy."
            )
        df[columns] = df[columns].fillna(fill_value)
    elif strategy == "ffill":
        df[columns] = df[columns].ffill()
    elif strategy == "bfill":
        df[columns] = df[columns].bfill()
    else:
        raise ValueError(f"Unsupported categorical strategy: {strategy}")


def _apply_datetime_strategy(
    df: pd.DataFrame,
    columns: Sequence[str],
    strategy: Optional[DatetimeStrategy],
    fill_value: Optional[Any],
    indicator: bool,
) -> None:
    if not columns or strategy is None:
        return

    mask = df[columns].isna()
    if indicator:
        for column in columns:
            _set_indicator(df, column, mask[column])

    if strategy == "ffill":
        df[columns] = df[columns].ffill()
    elif strategy == "bfill":
        df[columns] = df[columns].bfill()
    elif strategy == "interpolate":
        df[columns] = df[columns].apply(lambda series: series.interpolate(limit_direction="both"))
    elif strategy == "constant":
        if fill_value is None:
            raise ValueError(
                "datetime_fill_value is required when using the 'constant' strategy."
            )
        df[columns] = df[columns].fillna(fill_value)
    else:
        raise ValueError(f"Unsupported datetime strategy: {strategy}")


def _select_columns(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]],
    *,
    include_dtypes: Iterable[str],
    exclude: Iterable[str],
) -> Sequence[str]:
    exclude_set = set(exclude)
    if columns is not None:
        return [column for column in columns if column in df.columns and column not in exclude_set]

    dtype_frame = df.select_dtypes(include=include_dtypes)
    return [column for column in dtype_frame.columns if column not in exclude_set]


def _set_indicator(df: pd.DataFrame, column: str, mask: pd.Series) -> None:
    """Attach or update an indicator column for imputations."""

    indicator_name = f"{column}_was_missing"
    if indicator_name in df.columns:
        df[indicator_name] = df[indicator_name] | mask
    else:
        df[indicator_name] = mask


__all__ = [
    "CONFIG_PATH",
    "MissingValueConfig",
    "DATASET_MISSING_VALUE_PRESETS",
    "get_missing_value_config",
    "handle_missing_values",
]
