"""Configuration-driven preprocessing and experiment utilities."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import get_section
from .feature_engineering import apply_feature_steps
from .missing_values import get_missing_value_config, handle_missing_values
from .outliers import fit_outlier_clip_model, transform_with_clip_model
from .rolling_features import apply_rolling_features
from .scaling import fit_scaler_model, transform_with_scaler

LOG_CLIP_LOWER = -0.999999
DEFAULT_GROUPBY = ("Ticker", "Date")


def get_preprocessing_config(dataset_key: str) -> Mapping[str, object]:
    preprocessing = get_section("preprocessing")
    if dataset_key not in preprocessing:
        raise KeyError(f"Preprocessing config not found for dataset '{dataset_key}'")
    return copy.deepcopy(preprocessing[dataset_key])


def apply_base_preprocessing(
    df: pd.DataFrame,
    dataset_key: str,
    *,
    profile_name: str | None = None,
) -> tuple[pd.DataFrame, MutableMapping[str, object]]:
    """Apply missing value handling, transforms, and rolling features.

    Returns the transformed dataframe along with the effective preprocessing
    configuration (after resolving any profile overrides).
    """

    config = _resolve_profile_config(dataset_key, profile_name)
    missing_override = config.pop("missing_override", None)
    transforms = config.get("transforms") or []

    working = df.copy()
    order_override = config.get("order_columns")
    if order_override:
        order_columns = [column for column in order_override if column in working.columns]
    else:
        order_columns = [column for column in DEFAULT_GROUPBY if column in working.columns]
    if order_columns:
        working = working.sort_values(order_columns).reset_index(drop=True)

    preset_name = config.get("missing_values_preset") or dataset_key
    missing_config = get_missing_value_config(preset_name)
    if missing_override:
        missing_config = replace(missing_config, **missing_override)
    working = handle_missing_values(working, config=missing_config)

    working = _apply_transforms(working, transforms)
    working = apply_rolling_features(working, config.get("rolling_features"))

    return working, config


def apply_feature_sets(
    df: pd.DataFrame,
    dataset_key: str,
    feature_set_names: Iterable[str],
    *,
    config_override: Optional[Mapping[str, object]] = None,
) -> pd.DataFrame:
    config = copy.deepcopy(config_override) if config_override is not None else get_preprocessing_config(dataset_key)
    feature_sets = config.get("feature_sets") or {}

    working = df.copy()
    for name in feature_set_names:
        if name not in feature_sets:
            raise KeyError(f"Feature set '{name}' not defined for dataset '{dataset_key}'")
        steps = feature_sets[name].get("steps")
        working = apply_feature_steps(working, steps)
    return working


def append_target(
    df: pd.DataFrame,
    dataset_key: str,
    target_column: str = "target",
    *,
    config_override: Optional[Mapping[str, object]] = None,
) -> pd.DataFrame:
    config = copy.deepcopy(config_override) if config_override is not None else get_preprocessing_config(dataset_key)
    target_cfg = config.get("target")
    if not target_cfg:
        raise KeyError(f"Target definition missing in config for dataset '{dataset_key}'")

    target_type = target_cfg.get("type") or "direction"
    price_column = target_cfg.get("price_column") or "Close"
    horizon = int(target_cfg.get("horizon", 1))
    positive_label = target_cfg.get("positive_label", 1)
    negative_label = target_cfg.get("negative_label", 0)
    groupby = target_cfg.get("groupby")

    working = df.copy()

    if target_type == "direction":
        if groupby:
            grouped_price = working.groupby(list(groupby))[price_column]
            future_price = grouped_price.shift(-horizon)
        else:
            future_price = working[price_column].shift(-horizon)

        label = (future_price > working[price_column]).astype(int)
        label = label.where(~label.isna(), other=negative_label)

        working[target_column] = label
    elif target_type == "column":
        label_column = target_cfg.get("column")
        if not label_column:
            raise KeyError("Target configuration with type 'column' requires a 'column' entry.")
        working[target_column] = working[label_column]
    else:
        raise ValueError(f"Unsupported target type: {target_type}")

    working = working.dropna(subset=[target_column])
    return working


def select_feature_columns(
    df: pd.DataFrame,
    dataset_key: str,
    target_column: str = "target",
    *,
    config_override: Optional[Mapping[str, object]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    config = copy.deepcopy(config_override) if config_override is not None else get_preprocessing_config(dataset_key)
    selection_cfg = config.get("feature_selection") or {}
    exclude = set(selection_cfg.get("exclude", [])) | {target_column}

    feature_columns = [
        column
        for column in df.columns
        if column not in exclude and pd.api.types.is_numeric_dtype(df[column])
    ]

    features = df[feature_columns]
    target = df[target_column]
    return features, target


def apply_post_split_transforms(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    profile_config: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clip_config = profile_config.get("outlier_clip")
    if clip_config:
        clip_model = fit_outlier_clip_model(train_df, clip_config)
        if clip_model:
            train_df = transform_with_clip_model(train_df, clip_model)
            test_df = transform_with_clip_model(test_df, clip_model)

    scaling_config = profile_config.get("scaling")
    if scaling_config and scaling_config.get("strategy") not in (None, "none"):
        scaler_model = fit_scaler_model(train_df, scaling_config)
        if scaler_model:
            train_df = transform_with_scaler(train_df, scaler_model)
            test_df = transform_with_scaler(test_df, scaler_model)

    return train_df, test_df


def get_experiment_config(experiment_key: str) -> Mapping[str, object]:
    experiments = get_section("experiments")
    if experiment_key not in experiments:
        raise KeyError(f"Experiment '{experiment_key}' not defined in configuration")
    return copy.deepcopy(experiments[experiment_key])


def _resolve_profile_config(dataset_key: str, profile_name: str | None) -> MutableMapping[str, object]:
    config = get_preprocessing_config(dataset_key)
    profiles = config.pop("profiles", {})
    if profile_name is None:
        return config

    if profile_name not in profiles:
        raise KeyError(f"Unknown preprocessing profile '{profile_name}' for dataset '{dataset_key}'")

    overrides = profiles[profile_name]
    merged = _deep_merge_dicts(config, overrides)
    merged.pop("profiles", None)
    return merged


def _deep_merge_dicts(
    base: MutableMapping[str, object],
    override: Mapping[str, object],
) -> MutableMapping[str, object]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dicts(result[key], value)
        elif isinstance(value, list):
            result[key] = copy.deepcopy(value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _apply_transforms(
    df: pd.DataFrame,
    transforms: Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    if not transforms:
        return df

    working = df.copy()
    for transform in transforms:
        step_type = (transform.get("type") or "").lower()
        if step_type == "log1p_safe":
            columns = _valid_columns(working, transform.get("columns"))
            for column in columns:
                working[column] = _log1p_safe(working[column])
        elif step_type == "signed_log1p":
            columns = _valid_columns(working, transform.get("columns"))
            for column in columns:
                working[column] = _signed_log1p(working[column])
        elif step_type == "returns":
            columns = _valid_columns(working, transform.get("columns"))
            lags = transform.get("lags") or [1]
            groupby = transform.get("groupby")
            prefix = transform.get("prefix") or "ret"
            working = _add_returns(working, columns, lags, groupby, prefix)
        else:
            raise ValueError(f"Unsupported transform type: {step_type}")

    return working


def _add_returns(
    df: pd.DataFrame,
    columns: Sequence[str],
    lags: Sequence[int],
    groupby: Sequence[str] | None,
    prefix: str,
) -> pd.DataFrame:
    working = df.copy()
    groupby_list = list(groupby) if groupby else None

    for column in columns:
        if groupby_list:
            grouped = working.groupby(groupby_list)[column]
            for lag in lags:
                name = f"{prefix}_{column}_pct_change_{lag}"
                working[name] = grouped.transform(lambda series: series.pct_change(periods=lag))
        else:
            for lag in lags:
                name = f"{prefix}_{column}_pct_change_{lag}"
                working[name] = working[column].pct_change(periods=lag)

    new_columns = [
        f"{prefix}_{column}_pct_change_{lag}"
        for column in columns
        for lag in lags
    ]
    working[new_columns] = working[new_columns].replace([np.inf, -np.inf], np.nan)
    return working


def _valid_columns(df: pd.DataFrame, columns: Sequence[str] | None) -> list[str]:
    if not columns:
        return []
    return [column for column in columns if column in df.columns]


def _log1p_safe(series: pd.Series) -> pd.Series:
    sanitized = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sanitized = sanitized.clip(lower=LOG_CLIP_LOWER)
    return np.log1p(sanitized)


def _signed_log1p(series: pd.Series) -> pd.Series:
    sanitized = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return np.sign(sanitized) * np.log1p(np.abs(sanitized))


__all__ = [
    "apply_base_preprocessing",
    "apply_feature_sets",
    "append_target",
    "select_feature_columns",
    "apply_post_split_transforms",
    "get_preprocessing_config",
    "get_experiment_config",
]
