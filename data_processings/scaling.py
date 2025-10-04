"""Scaling models that fit on training data and transform splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd


@dataclass
class ScalingModel:
    strategy: str
    columns: list[str]
    groupby: Optional[Tuple[str, ...]]
    params: Dict[Tuple, Dict[str, Dict[str, float]]]
    global_params: Dict[str, Dict[str, float]]
    feature_range: Tuple[float, float] | None = None


def fit_scaler_model(
    train_df: pd.DataFrame,
    config: Mapping[str, object] | None,
) -> Optional[ScalingModel]:
    if not config:
        return None

    strategy = config.get("strategy")
    if strategy is None or str(strategy).lower() in {"none", ""}:
        return None
    strategy = str(strategy).lower()

    columns = _resolve_columns(train_df, config.get("columns"))
    if not columns:
        return None

    groupby = config.get("groupby")
    if isinstance(groupby, (list, tuple)) and len(groupby) == 0:
        groupby = None
    if groupby:
        groupby_tuple = tuple(groupby)
    else:
        groupby_tuple = None

    feature_range = None
    if strategy == "minmax":
        feature_range = tuple(config.get("feature_range", [0.0, 1.0]))  # type: ignore[arg-type]

    global_params = _compute_params(train_df, columns, strategy)

    group_params: Dict[Tuple, Dict[str, Dict[str, float]]] = {}
    if groupby_tuple:
        for key, group in train_df.groupby(list(groupby_tuple)):
            key_tuple = key if isinstance(key, tuple) else (key,)
            group_params[key_tuple] = _compute_params(group, columns, strategy)

    return ScalingModel(
        strategy=strategy,
        columns=columns,
        groupby=groupby_tuple,
        params=group_params,
        global_params=global_params,
        feature_range=feature_range,
    )


def transform_with_scaler(df: pd.DataFrame, model: Optional[ScalingModel]) -> pd.DataFrame:
    if model is None:
        return df

    working = df.copy()
    columns = [column for column in model.columns if column in working.columns]
    if not columns:
        return working

    if model.groupby:
        grouped = working.groupby(list(model.groupby))
        for key, indices in grouped.groups.items():
            key_tuple = key if isinstance(key, tuple) else (key,)
            params = model.params.get(key_tuple, model.global_params)
            working.loc[indices, columns] = _apply_strategy(
                working.loc[indices, columns],
                model.strategy,
                params,
                model.feature_range,
            )
    else:
        working[columns] = _apply_strategy(
            working[columns],
            model.strategy,
            model.global_params,
            model.feature_range,
        )

    return working


def _resolve_columns(df: pd.DataFrame, columns: Sequence[str] | None) -> list[str]:
    if columns is None:
        return df.select_dtypes(include=["number"]).columns.tolist()
    return [column for column in columns if column in df.columns]


def _compute_params(df: pd.DataFrame, columns: Sequence[str], strategy: str) -> Dict[str, Dict[str, float]]:
    numeric_df = df[columns]
    if strategy == "standard":
        mean = numeric_df.mean()
        std = numeric_df.std(ddof=0).replace(0, 1.0).fillna(1.0)
        return {"center": mean.to_dict(), "scale": std.to_dict()}

    if strategy == "minmax":
        min_val = numeric_df.min()
        max_val = numeric_df.max()
        return {"min": min_val.to_dict(), "max": max_val.to_dict()}

    if strategy == "robust":
        median = numeric_df.median()
        q75 = numeric_df.quantile(0.75)
        q25 = numeric_df.quantile(0.25)
        iqr = (q75 - q25).replace(0, 1.0).fillna(1.0)
        return {"center": median.to_dict(), "scale": iqr.to_dict()}

    raise ValueError(f"Unsupported scaling strategy: {strategy}")


def _apply_strategy(
    values: pd.DataFrame,
    strategy: str,
    params: Mapping[str, Dict[str, float]],
    feature_range: Tuple[float, float] | None,
) -> pd.DataFrame:
    if strategy == "standard":
        center = pd.Series(params["center"])  # type: ignore[index]
        scale = pd.Series(params["scale"])  # type: ignore[index]
        scale = scale.replace(0, 1.0).fillna(1.0)
        return (values - center) / scale

    if strategy == "minmax":
        lower, upper = feature_range if feature_range is not None else (0.0, 1.0)
        min_val = pd.Series(params["min"])  # type: ignore[index]
        max_val = pd.Series(params["max"])  # type: ignore[index]
        scale = (max_val - min_val).replace(0, 1.0).fillna(1.0)
        scaled = (values - min_val) / scale
        return scaled * (upper - lower) + lower

    if strategy == "robust":
        center = pd.Series(params["center"])  # type: ignore[index]
        scale = pd.Series(params["scale"])  # type: ignore[index]
        scale = scale.replace(0, 1.0).fillna(1.0)
        return (values - center) / scale

    raise ValueError(f"Unsupported scaling strategy: {strategy}")


__all__ = ["fit_scaler_model", "transform_with_scaler", "ScalingModel"]
