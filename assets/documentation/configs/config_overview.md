# `assets/config.json` Overview

This document describes the top-level structure of `assets/config.json` and how the individual sections interact. Dedicated deep dives for each major section live in the accompanying files:

- [`preprocessing_config.md`](./preprocessing_config.md) – Dataset-specific preprocessing configuration.
- [`models_config.md`](./models_config.md) – Traditional ML model registry and neural model definitions.
- [`experiments_config.md`](./experiments_config.md) – Experiment entries that tie datasets, preprocessing, and models together.

## Top-Level Keys

| Key | Purpose |
| --- | --- |
| `missing_value_presets` | Shared imputation configurations referenced by multiple datasets. |
| `preprocessing` | Dataset-specific preprocessing templates and optional profile overrides. |
| `models` | Registry of traditional scikit-learn classifiers instantiated via `models/registry.py`. |
| `deep_models` | Neural-network definitions consumed by `models/neural.py`. |
| `experiments` | Experiment blocks that describe datasets, preprocessing axes, models, and metrics. |

Each subsection inherits JSON objects that can be overridden or extended for new datasets/experiments. See the linked documents for detailed field descriptions and usage examples.
