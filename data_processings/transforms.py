"""Composable DataFrame transforms built on the DFX pipeline foundation."""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .feature_engineering import apply_feature_steps
from .missing_values import get_missing_value_config, handle_missing_values
from .outliers import OutlierClipModel, fit_outlier_clip_model, transform_with_clip_model
from .rolling_features import apply_rolling_features
from .scaling import ScalingModel, fit_scaler_model, transform_with_scaler

# ---------------------------------------------------------------------------
# Column specification helpers
# ---------------------------------------------------------------------------

ColSpec = Union[None, str, Sequence[str], re.Pattern, Callable[[pd.DataFrame], Sequence[str]]]


def _resolve_columns(df: pd.DataFrame, spec: ColSpec) -> List[str]:
    """Resolve flexible column specifications into a concrete ordered list."""
    if spec is None:
        return list(df.columns)
    if isinstance(spec, str):
        return [spec] if spec in df.columns else []
    if isinstance(spec, re.Pattern):
        return [column for column in df.columns if spec.search(column)]
    if callable(spec):
        resolved = spec(df)
        return [column for column in resolved if column in df.columns]
    return [column for column in spec if column in df.columns]


_STAGE_ALIASES: Mapping[str, str] = {
    "pre": "global",
    "post": "train_test",
    "both": "train_test",
    "train": "train_only",
    "test": "test_only",
}

_STAGE_MAP: Mapping[str, set[str]] = {
    "global": {"global", "all"},
    "train": {"train", "train_only", "train_test", "both", "all"},
    "test": {"test", "test_only", "train_test", "both", "all"},
}


# ---------------------------------------------------------------------------
# Base transform
# ---------------------------------------------------------------------------


@dataclass
class DFXTransform(abc.ABC):
    """Mother class for all DataFrame transforms with sklearn-like API."""

    name: str
    cols: ColSpec = None
    stage: str = "train_test"
    seed: Optional[int] = None
    keep_unselected: bool = True
    strict_schema: bool = False
    enabled: bool = True

    _fitted: bool = field(init=False, default=False)
    _selected_cols: List[str] = field(init=False, default_factory=list)
    _lineage: Dict[str, List[str]] = field(init=False, default_factory=dict)

    def applies_to_stage(self, stage: str) -> bool:
        normalized = _STAGE_ALIASES.get(self.stage, self.stage)
        target = _STAGE_MAP.get(stage)
        if target is None:
            raise ValueError(f"Unknown pipeline stage '{stage}'.")
        return normalized in target

    # -------------------- sklearn-like API --------------------
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DFXTransform":
        Xsel = self._validate_and_select(X)
        self._rng = np.random.default_rng(self.seed)  # type: ignore[attr-defined]
        self._fit_impl(Xsel, y)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X
        if not self._fitted and self._requires_fit() and self.applies_to_stage("train"):
            raise RuntimeError(f"Transform '{self.name}' requires fit() before transform().")
        Xsel = self._select_for_transform(X)
        Xout, lineage = self._transform_impl(Xsel)
        self._lineage = lineage
        if self.keep_unselected:
            unselected = [column for column in X.columns if column not in Xsel.columns]
            merged = pd.concat([X[unselected], Xout], axis=1)
            return merged.loc[X.index]
        return Xout.loc[X.index]

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        if not self.enabled:
            return X
        Xsel = self._validate_and_select(X)
        self._rng = np.random.default_rng(self.seed)  # type: ignore[attr-defined]
        if self._requires_fit():
            self._fit_impl(Xsel, y)
            self._fitted = True
        Xout, lineage = self._transform_impl(Xsel)
        self._lineage = lineage
        if self.keep_unselected:
            unselected = [column for column in X.columns if column not in Xsel.columns]
            merged = pd.concat([X[unselected], Xout], axis=1)
            return merged.loc[X.index]
        return Xout.loc[X.index]

    # -------------------- extension points --------------------
    def _requires_fit(self) -> bool:
        return True

    def _validate_and_select(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        selected = _resolve_columns(X, self.cols)
        if self.strict_schema:
            missing = [column for column in selected if column not in X.columns]
            if missing:
                raise KeyError(f"Columns not in DataFrame: {missing}")
        if not selected:
            selected = []
        self._selected_cols = list(selected)
        if not selected:
            return X.iloc[:, 0:0].copy()
        return X.loc[:, selected].copy()

    def _select_for_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.cols is None:
            selected = self._selected_cols or list(X.columns)
        else:
            selected = _resolve_columns(X, self.cols)
        selected = [column for column in selected if column in X.columns]
        if not selected:
            return X.iloc[:, 0:0].copy()
        return X.loc[:, selected].copy()

    @abc.abstractmethod
    def _fit_impl(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        ...

    @abc.abstractmethod
    def _transform_impl(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        ...

    # diagnostics -----------------------------------------------------------------
    def lineage(self) -> Dict[str, List[str]]:
        return dict(self._lineage)

    def get_params(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_") and key not in {"_rng"}
        }

    def set_params(self, **kwargs: Any) -> "DFXTransform":
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"Unknown parameter: {key}")
            setattr(self, key, value)
        return self


# ---------------------------------------------------------------------------
# Helper utilities used by multiple transforms
# ---------------------------------------------------------------------------


def _identity_lineage(columns: Iterable[str]) -> Dict[str, List[str]]:
    return {column: [column] for column in columns}


def _log1p_safe(series: pd.Series) -> pd.Series:
    sanitized = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sanitized = sanitized.clip(lower=-0.999999)
    return np.log1p(sanitized)


def _signed_log1p(series: pd.Series) -> pd.Series:
    sanitized = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return np.sign(sanitized) * np.log1p(np.abs(sanitized))


def _add_returns(
    df: pd.DataFrame,
    columns: Sequence[str],
    lags: Sequence[int],
    groupby: Sequence[str] | None,
    prefix: str,
) -> pd.DataFrame:
    working = df.copy()
    group_cols = list(groupby) if groupby else None

    for column in columns:
        if column not in working.columns:
            continue
        if group_cols:
            grouped = working.groupby(group_cols)[column]
            for lag in lags:
                name = f"{prefix}_{column}_pct_change_{lag}"
                working[name] = grouped.transform(lambda series: series.pct_change(periods=lag))
        else:
            for lag in lags:
                name = f"{prefix}_{column}_pct_change_{lag}"
                working[name] = working[column].pct_change(periods=lag)

    return working.replace([np.inf, -np.inf], np.nan)


def _oversample_dataframe(df: pd.DataFrame, target_col: str, random_state: int) -> pd.DataFrame:
    counts = df[target_col].value_counts()
    if counts.empty:
        return df

    max_count = counts.max()
    rng = np.random.default_rng(random_state)
    frames: List[pd.DataFrame] = []
    for label, count in counts.items():
        subset = df[df[target_col] == label]
        if count < max_count:
            extra = subset.sample(
                max_count - count,
                replace=True,
                random_state=rng.integers(0, 2**32 - 1),
            )
            subset = pd.concat([subset, extra], axis=0)
        frames.append(subset)

    balanced = pd.concat(frames, axis=0).sample(frac=1.0, random_state=random_state)
    return balanced.reset_index(drop=True)


def _undersample_dataframe(df: pd.DataFrame, target_col: str, random_state: int) -> pd.DataFrame:
    counts = df[target_col].value_counts()
    if counts.empty:
        return df

    min_count = counts.min()
    rng = np.random.default_rng(random_state)
    frames: List[pd.DataFrame] = []
    for label, count in counts.items():
        subset = df[df[target_col] == label]
        if count > min_count:
            subset = subset.sample(
                min_count,
                replace=False,
                random_state=rng.integers(0, 2**32 - 1),
            )
        frames.append(subset)

    balanced = pd.concat(frames, axis=0).sample(frac=1.0, random_state=random_state)
    return balanced.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Concrete transforms
# ---------------------------------------------------------------------------


class SortingTransform(DFXTransform):
    order: Sequence[str]

    def __init__(self, name: str, order: Sequence[str], ascending: Optional[Sequence[bool]] = None, **kwargs: Any) -> None:
        super().__init__(name=name, cols=None, **kwargs)
        self.order = list(order)
        self.ascending = list(ascending) if ascending is not None else None

    def _requires_fit(self) -> bool:
        return False

    def _fit_impl(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        pass

    def _transform_impl(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        columns = [column for column in self.order if column in X.columns]
        if not columns:
            return X.copy(), _identity_lineage(X.columns)
        ascending = self.ascending if self.ascending is not None else True
        sorted_df = X.sort_values(columns, ascending=ascending).reset_index(drop=True)
        return sorted_df, _identity_lineage(sorted_df.columns)


class MissingValueTransform(DFXTransform):
    def __init__(
        self,
        name: str,
        preset: Optional[str] = None,
        override: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        stage = kwargs.pop("stage", "train_test")
        super().__init__(name=name, cols=None, stage=stage, **kwargs)
        self.preset = preset
        self.override = dict(override) if override else None

    def _requires_fit(self) -> bool:
        return False

    def _fit_impl(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        pass

    def _transform_impl(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        config = None
        if self.preset:
            config = get_missing_value_config(self.preset)
            if self.override:
                config = replace(config, **self.override)  # type: ignore[arg-type]
        processed = handle_missing_values(X, config=config)
        return processed, _identity_lineage(processed.columns)


class ConstantFillTransform(DFXTransform):
    def __init__(self, name: str, values: Mapping[str, Any], indicator: bool = False, **kwargs: Any) -> None:
        super().__init__(name=name, cols=None, **kwargs)
        self.values = dict(values)
        self.indicator = indicator

    def _requires_fit(self) -> bool:
        return False

    def _fit_impl(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        pass

    def _transform_impl(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        df = X.copy()
        for column, value in self.values.items():
            if column not in df.columns:
                continue
            mask = df[column].isna()
            if self.indicator:
                df[f"{column}_was_missing"] = mask
            df[column] = df[column].fillna(value)
        return df, _identity_lineage(df.columns)


class LogTransform(DFXTransform):
    def __init__(self, name: str, columns: Sequence[str], signed: bool = False, **kwargs: Any) -> None:
        super().__init__(name=name, cols=columns, **kwargs)
        self.signed = signed

    def _requires_fit(self) -> bool:
        return False

    def _fit_impl(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        pass

    def _transform_impl(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        Z = X.copy()
        lineage: Dict[str, List[str]] = {}
        for column in Z.columns:
            if self.signed:
                Z[column] = _signed_log1p(Z[column].astype(float))
            else:
                Z[column] = _log1p_safe(Z[column].astype(float))
            lineage[column] = [column]
        return Z, lineage


class ReturnsTransform(DFXTransform):
    def __init__(
        self,
        name: str,
        columns: Sequence[str],
        lags: Sequence[int],
        groupby: Optional[Sequence[str]] = None,
        prefix: str = "ret",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, cols=None, **kwargs)
        self.columns = list(columns)
        self.lags = list(lags)
        self.groupby = list(groupby) if groupby else None
        self.prefix = prefix

    def _requires_fit(self) -> bool:
        return False

    def _fit_impl(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        pass

    def _transform_impl(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        enriched = _add_returns(X, self.columns, self.lags, self.groupby, self.prefix)
        return enriched, _identity_lineage(enriched.columns)


class RollingTransform(DFXTransform):
    def __init__(self, name: str, specs: Sequence[Mapping[str, Any]], **kwargs: Any) -> None:
        super().__init__(name=name, cols=None, **kwargs)
        self.specs = list(specs)

    def _requires_fit(self) -> bool:
        return False

    def _fit_impl(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        pass

    def _transform_impl(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        if not self.specs:
            return X.copy(), _identity_lineage(X.columns)
        transformed = apply_rolling_features(X, self.specs)
        return transformed, _identity_lineage(transformed.columns)


class FeatureSetTransform(DFXTransform):
    def __init__(self, name: str, steps: Sequence[Mapping[str, Any]], **kwargs: Any) -> None:
        super().__init__(name=name, cols=None, **kwargs)
        self.steps = list(steps)

    def _requires_fit(self) -> bool:
        return False

    def _fit_impl(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        pass

    def _transform_impl(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        if not self.steps:
            return X.copy(), _identity_lineage(X.columns)
        transformed = apply_feature_steps(X, self.steps)
        return transformed, _identity_lineage(transformed.columns)


class TargetTransform(DFXTransform):
    def __init__(
        self,
        name: str,
        target_column: str = "target",
        target_type: str = "direction",
        price_column: str = "Close",
        horizon: int = 1,
        positive_label: int = 1,
        negative_label: int = 0,
        source_column: Optional[str] = None,
        groupby: Optional[Sequence[str]] = None,
        drop_na: bool = True,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("stage", "global")
        super().__init__(name=name, cols=None, **kwargs)
        self.target_column = target_column
        self.target_type = target_type
        self.price_column = price_column
        self.horizon = int(horizon)
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.source_column = source_column
        self.groupby = list(groupby) if groupby else None
        self.drop_na = drop_na

    def _requires_fit(self) -> bool:
        return False

    def _fit_impl(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        pass

    def _transform_impl(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        df = X.copy()
        if self.target_type == "direction":
            price_col = self.price_column
            if price_col not in df.columns:
                raise KeyError(f"Price column '{price_col}' not found for target transform '{self.name}'.")
            if self.groupby:
                grouped_price = df.groupby(self.groupby)[price_col]
                future_price = grouped_price.shift(-self.horizon)
            else:
                future_price = df[price_col].shift(-self.horizon)
            label = (future_price > df[price_col]).astype(int)
            label = label.where(~label.isna(), other=self.negative_label)
            df[self.target_column] = label
        elif self.target_type == "column":
            column = self.source_column or self.price_column
            if column not in df.columns:
                raise KeyError(f"Target column '{column}' not present in DataFrame.")
            df[self.target_column] = df[column]
        else:
            raise ValueError(f"Unsupported target type '{self.target_type}'.")

        if self.drop_na:
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[self.target_column])
        return df, _identity_lineage(df.columns)


class DropColumnsTransform(DFXTransform):
    def __init__(self, name: str, columns: Sequence[str], **kwargs: Any) -> None:
        super().__init__(name=name, cols=None, **kwargs)
        self.columns = list(columns)

    def _requires_fit(self) -> bool:
        return False

    def _fit_impl(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        pass

    def _transform_impl(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        existing = [column for column in self.columns if column in X.columns]
        if not existing:
            return X.copy(), _identity_lineage(X.columns)
        pruned = X.drop(columns=existing)
        return pruned, _identity_lineage(pruned.columns)


class ClassBalanceTransform(DFXTransform):
    def __init__(
        self,
        name: str,
        method: str,
        target_column: str,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("stage", "train_only")
        super().__init__(name=name, cols=None, **kwargs)
        self.method = method.lower()
        self.target_column = target_column
        self.random_state = int(random_state)

    def _requires_fit(self) -> bool:
        return False

    def _fit_impl(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        pass

    def _transform_impl(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        if self.target_column not in X.columns:
            raise KeyError(f"Target column '{self.target_column}' not found for balancing.")
        if self.method == "oversample":
            balanced = _oversample_dataframe(X, self.target_column, self.random_state)
        elif self.method == "undersample":
            balanced = _undersample_dataframe(X, self.target_column, self.random_state)
        else:
            raise ValueError(f"Unsupported class balance method: {self.method}")
        return balanced, _identity_lineage(balanced.columns)


class OutlierClipTransform(DFXTransform):
    def __init__(
        self,
        name: str,
        *,
        stage: str = "train_test",
        cols: ColSpec = None,
        seed: Optional[int] = None,
        keep_unselected: bool = True,
        strict_schema: bool = False,
        enabled: bool = True,
        **fit_params: Any,
    ) -> None:
        super().__init__(
            name=name,
            cols=cols,
            stage=stage,
            seed=seed,
            keep_unselected=keep_unselected,
            strict_schema=strict_schema,
            enabled=enabled,
        )
        self.fit_params = dict(fit_params)
        self.clip_model: Optional[OutlierClipModel] = None

    def _requires_fit(self) -> bool:
        return True

    def _fit_impl(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        params = dict(self.fit_params)
        self.clip_model = fit_outlier_clip_model(X, params)

    def _transform_impl(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        if self.clip_model is None:
            return X.copy(), _identity_lineage(X.columns)
        transformed = transform_with_clip_model(X, self.clip_model)
        return transformed, _identity_lineage(transformed.columns)


class ScalingTransform(DFXTransform):
    def __init__(
        self,
        name: str,
        *,
        stage: str = "train_test",
        cols: ColSpec = None,
        seed: Optional[int] = None,
        keep_unselected: bool = True,
        strict_schema: bool = False,
        enabled: bool = True,
        **fit_params: Any,
    ) -> None:
        super().__init__(
            name=name,
            cols=cols,
            stage=stage,
            seed=seed,
            keep_unselected=keep_unselected,
            strict_schema=strict_schema,
            enabled=enabled,
        )
        self.fit_params = dict(fit_params)
        self.scaler_model: Optional[ScalingModel] = None

    def _requires_fit(self) -> bool:
        return True

    def _fit_impl(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        params = dict(self.fit_params)
        self.scaler_model = fit_scaler_model(X, params)

    def _transform_impl(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        if self.scaler_model is None:
            return X.copy(), _identity_lineage(X.columns)
        transformed = transform_with_scaler(X, self.scaler_model)
        return transformed, _identity_lineage(transformed.columns)


class ColumnMapTransform(DFXTransform):
    def __init__(
        self,
        name: str,
        fn: Callable[[pd.Series], pd.Series],
        suffix: Optional[str] = None,
        in_place: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, cols=None if not kwargs.get("cols") else kwargs["cols"], **kwargs)
        self.fn = fn
        self.suffix = suffix
        self.in_place = in_place

    def _requires_fit(self) -> bool:
        return False

    def _fit_impl(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        pass

    def _transform_impl(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        Z = X.copy()
        lineage: Dict[str, List[str]] = {}
        for column in X.columns:
            out_col = column if self.in_place else f"{column}{self.suffix or '_map'}"
            Z[out_col] = self.fn(X[column])
            lineage[column] = [out_col]
        if self.in_place:
            return Z, _identity_lineage(Z.columns)
        return Z, lineage


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class DFXPipeline:
    def __init__(self, steps: Sequence[DFXTransform], name: str = "dfx_pipeline") -> None:
        self.steps = list(steps)
        self.name = name

    def apply_global(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self._run(X, stage="global", y=y)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DFXPipeline":
        self._run(X, stage="train", y=y)
        return self

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self._run(X, stage="train", y=y)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._run(X, stage="test")

    def summary(self) -> str:
        lines = [f"DFXPipeline(name={self.name}, steps={len(self.steps)})"]
        for index, transform in enumerate(self.steps):
            params = transform.get_params()
            lines.append(f"  {index:02d}. {transform.__class__.__name__}({params})")
        return "\n".join(lines)

    # internal -------------------------------------------------------------------
    def _run(self, X: pd.DataFrame, stage: str, y: Optional[pd.Series] = None) -> pd.DataFrame:
        data = X
        for transform in self.steps:
            if not transform.enabled:
                continue
            if not transform.applies_to_stage(stage):
                continue
            if stage in ("global", "train"):
                data = transform.fit_transform(data, y)
            else:
                data = transform.transform(data)
        return data


# Backwards-compatible alias -------------------------------------------------

TransformPipeline = DFXPipeline

# Registry mapping -----------------------------------------------------------

TRANSFORM_REGISTRY: Mapping[str, Callable[..., DFXTransform]] = {
    "sorting": SortingTransform,
    "missing_values": MissingValueTransform,
    "constant_fill": ConstantFillTransform,
    "log1p": LogTransform,
    "signed_log1p": lambda **kwargs: LogTransform(signed=True, **kwargs),
    "returns": ReturnsTransform,
    "rolling": RollingTransform,
    "feature_set": FeatureSetTransform,
    "target": TargetTransform,
    "drop_columns": DropColumnsTransform,
    "class_balance": ClassBalanceTransform,
    "outlier_clip": OutlierClipTransform,
    "scaling": ScalingTransform,
    "column_map": ColumnMapTransform,
}


def build_transform(name: str, cfg: Mapping[str, Any]) -> Optional[DFXTransform]:
    ttype = cfg.get("type")
    if not ttype:
        raise KeyError(f"Transform '{name}' missing 'type'.")
    factory = TRANSFORM_REGISTRY.get(str(ttype))
    if factory is None:
        raise KeyError(f"Unknown transform type '{ttype}' in '{name}'.")

    params = dict(cfg.get("params", {}))
    params.setdefault("name", name)

    enabled = bool(cfg.get("enabled", True))
    params.setdefault("enabled", enabled)

    stage = cfg.get("stage")
    apply_to = cfg.get("apply_to")
    if stage is None and apply_to is not None:
        stage = _STAGE_ALIASES.get(str(apply_to), str(apply_to))
    if stage is not None:
        params.setdefault("stage", stage)

    if "cols" in cfg:
        params.setdefault("cols", cfg["cols"])

    transform = factory(**params)
    if not transform.enabled:
        return None
    return transform


__all__ = [
    "DFXPipeline",
    "DFXTransform",
    "TransformPipeline",
    "build_transform",
    "TRANSFORM_REGISTRY",
]
