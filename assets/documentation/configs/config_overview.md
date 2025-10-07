# `assets/config.json` Overview

`assets/config.json` now acts as a slim registry of shared presets that are reused across all experiment profiles. Per-question configurations (dataset options, transform pipelines, model lists) live under `assets/configs/q1|q2`. This document summarises the active top-level keys; consult the linked deep dives for details.

- [`preprocessing_config.md`](./preprocessing_config.md) – Structure of the per-profile JSON files.
- [`models_config.md`](./models_config.md) – Traditional and neural model schema examples.
- [`experiments_config.md`](./experiments_config.md) – How profiles bundle datasets, transforms, and models.

## Top-Level Keys

| Key | Purpose |
| --- | --- |
| `missing_value_presets` | Named imputation strategies (numeric/categorical/datetime) referenced by the `missing_values` transform. |
| `models` | scikit-learn classifiers instantiated via `models.build_model(model_key)`. |
| `deep_models` | PyTorch architectures and training defaults consumed by `models.neural`. |

### `missing_value_presets`
Each preset defines strategies for numeric, categorical, and datetime columns, along with optional drop thresholds and indicator behaviour. Example:

```json
"stock_market": {
  "numeric_strategy": "ffill",
  "categorical_strategy": "ffill",
  "drop_rows_threshold": 0.9,
  "indicator": true
}
```

Presets are retrieved with `data_processings.missing_values.get_missing_value_config(preset_name)` and injected into transform parameters:

```json
"missing": {
  "type": "missing_values",
  "stage": "global",
  "params": { "preset": "stock_market" }
}
```

### `models`
Traditional model entries describe how to import and instantiate estimators. The notebooks enumerate the keys specified in each profile’s `models` array.

```json
"random_forest": {
  "module": "sklearn.ensemble",
  "class": "RandomForestClassifier",
  "params": {
    "n_estimators": 200,
    "max_depth": null,
    "random_state": 42,
    "n_jobs": -1
  }
}
```

Override defaults at runtime with `build_model("random_forest", overrides={"max_depth": 12})` if needed.

### `deep_models`
Neural entries bundle architecture hyperparameters and training defaults. They are consumed by `models.neural.build_neural_model`.

```json
"mlp": {
  "module": "models.neural",
  "class": "TabularMLP",
  "params": { "hidden_dims": [128, 64], "dropout": 0.2 },
  "training": {
    "epochs": 20,
    "batch_size": 512,
    "learning_rate": 0.001,
    "grad_clip": 1.0
  }
}
```

Profiles list neural keys under `neural_models`; the Q2 notebook uses the training block when snapshots need to be generated.
