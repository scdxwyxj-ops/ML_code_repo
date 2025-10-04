"""Shared configuration loaders for preprocessing components."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

CONFIG_PATH = Path(__file__).resolve().parents[1] / "assets" / "config.json"


@lru_cache(maxsize=1)
def load_config() -> Mapping[str, Any]:
    with CONFIG_PATH.open() as config_file:
        return json.load(config_file)


def get_section(section: str) -> Any:
    config = load_config()
    if section not in config:
        raise KeyError(f"Configuration section '{section}' not found")
    return config[section]


__all__ = ["CONFIG_PATH", "load_config", "get_section"]
