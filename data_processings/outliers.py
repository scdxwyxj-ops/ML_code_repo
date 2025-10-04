"""Outlier clipping utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd


@dataclass
class OutlierClipModel:
    columns: list[str]
    global_lower: pd.Series
    global_upper: pd.Series
    groupby: Optional[Tuple[str, ...]]
    group_bounds: Dict[Tuple, Tuple[pd.Series, pd.Series]]


def fit_outlier_clip_model(
    train_df: pd.DataFrame,
    config: Mapping[str, object] | None,
) -> Optional[OutlierClipModel]:
    if not config:
        return None

    method = (config.get("method") or "").lower()
    if method != "winsorize":
        raise ValueError(f"Unsupported outlier clipping method: {method}")

    columns = [column for column in config.get("columns", []) if column in train_df.columns]
    if not columns:
        return None

    lower_q = float(config.get("lower", 0.01))
    upper_q = float(config.get("upper", 0.99))
    per_group = bool(config.get("per_group", False))
    groupby = config.get("groupby")

    if per_group and not groupby and "Ticker" in train_df.columns:
        groupby = ["Ticker"]
    if groupby:
        groupby_tuple = tuple(groupby)
    else:
        groupby_tuple = None

    global_lower = train_df[columns].quantile(lower_q)
    global_upper = train_df[columns].quantile(upper_q)

    group_bounds: Dict[Tuple, Tuple[pd.Series, pd.Series]] = {}
    if groupby_tuple:
        for key, group in train_df.groupby(list(groupby_tuple)):
            key_tuple = key if isinstance(key, tuple) else (key,)
            lower = group[columns].quantile(lower_q)
            upper = group[columns].quantile(upper_q)
            group_bounds[key_tuple] = (lower, upper)

    return OutlierClipModel(
        columns=columns,
        global_lower=global_lower,
        global_upper=global_upper,
        groupby=groupby_tuple,
        group_bounds=group_bounds,
    )


def transform_with_clip_model(
    df: pd.DataFrame,
    model: Optional[OutlierClipModel],
) -> pd.DataFrame:
    if model is None:
        return df

    working = df.copy()
    if not model.columns:
        return working

    if model.groupby:
        grouped = working.groupby(list(model.groupby))
        for key, indices in grouped.groups.items():
            key_tuple = key if isinstance(key, tuple) else (key,)
            lower, upper = model.group_bounds.get(key_tuple, (model.global_lower, model.global_upper))
            working.loc[indices, model.columns] = working.loc[indices, model.columns].clip(lower=lower, upper=upper, axis=1)
    else:
        working[model.columns] = working[model.columns].clip(lower=model.global_lower, upper=model.global_upper, axis=1)

    return working


__all__ = ["fit_outlier_clip_model", "transform_with_clip_model", "OutlierClipModel"]
