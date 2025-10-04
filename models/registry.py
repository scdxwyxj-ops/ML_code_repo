"""Model registry powered by assets/config.json."""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any, Mapping

from data_processings.config import get_section


@lru_cache(maxsize=1)
def _load_models_section() -> Mapping[str, Any]:
    return get_section("models")


def get_model_config(model_key: str) -> Mapping[str, Any]:
    models = _load_models_section()
    if model_key not in models:
        raise KeyError(f"Model '{model_key}' not defined in configuration")
    return models[model_key]


def build_model(model_key: str, overrides: Mapping[str, Any] | None = None) -> Any:
    config = dict(get_model_config(model_key))
    module_name = config.pop("module")
    class_name = config.pop("class")
    params = config.get("params", {}).copy()
    if overrides:
        params.update(overrides)

    module = importlib.import_module(module_name)
    model_cls = getattr(module, class_name)
    return model_cls(**params)


__all__ = ["build_model", "get_model_config"]
