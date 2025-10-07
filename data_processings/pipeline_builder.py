"""Utilities to build transform pipelines from structured configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

from .transforms import DFXPipeline, DFXTransform, build_transform


def load_pipeline_config(path: Path | str) -> Mapping[str, object]:
    with Path(path).open() as handle:
        return json.load(handle)


def build_pipeline_from_config(config: Mapping[str, object]) -> Tuple[DFXPipeline, Dict[str, object]]:
    transforms_cfg = config.get("transforms")
    pipeline_order = config.get("pipeline")
    if not isinstance(transforms_cfg, Mapping):
        raise ValueError("Config must contain a 'transforms' mapping.")
    if not isinstance(pipeline_order, Sequence):
        raise ValueError("Config must contain a 'pipeline' sequence.")

    transforms: list[DFXTransform] = []
    metadata: Dict[str, object] = {}

    for name in pipeline_order:
        if name not in transforms_cfg:
            raise KeyError(f"Transform '{name}' referenced in pipeline but not defined in 'transforms'.")
        cfg = transforms_cfg[name]
        if not isinstance(cfg, Mapping):
            raise ValueError(f"Transform '{name}' configuration must be a mapping.")
        transform = build_transform(name, cfg)
        if transform is not None:
            transforms.append(transform)

    metadata_keys = (
        "target_column",
        "drop_columns",
        "retain_columns",
        "label_column",
        "feature_allowlist",
        "feature_denylist",
    )
    for key in metadata_keys:
        if key in config:
            metadata[key] = config[key]

    return DFXPipeline(transforms), metadata
