"""Preprocessing package aggregating configuration-driven utilities."""

from .config import CONFIG_PATH, get_section, load_config
from .datasets import CreditCardDataset, StockMarketDataset, LendingClubDataset, load_credit_card_data, load_stock_market_data
from .feature_engineering import apply_feature_steps, process_q3_features
from .missing_values import (
    DATASET_MISSING_VALUE_PRESETS,
    MissingValueConfig,
    get_missing_value_config,
    handle_missing_values,
)
from .outliers import fit_outlier_clip_model, transform_with_clip_model
from .pipeline_builder import build_pipeline_from_config, load_pipeline_config
from .rolling_features import apply_rolling_features
from .scaling import fit_scaler_model, transform_with_scaler
from .transforms import DFXPipeline, DFXTransform, TransformPipeline, build_transform

__all__ = [
    "CONFIG_PATH",
    "DATASET_MISSING_VALUE_PRESETS",
    "MissingValueConfig",
    "apply_feature_steps",
    "apply_rolling_features",
    "fit_outlier_clip_model",
    "fit_scaler_model",
    "build_pipeline_from_config",
    "load_pipeline_config",
    "get_missing_value_config",
    "get_section",
    "handle_missing_values",
    "load_config",
    "CreditCardDataset",
    "StockMarketDataset",
    "LendingClubDataset",
    "load_credit_card_data",
    "load_stock_market_data",
    "transform_with_clip_model",
    "transform_with_scaler",
    "DFXPipeline",
    "DFXTransform",
    "TransformPipeline",
    "build_transform",
    "process_q3_features",
]
