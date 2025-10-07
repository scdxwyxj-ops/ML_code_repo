# Experiments & Profiles

Experiment orchestration now lives entirely inside the per-question profile files located at:

- `assets/configs/q1/*.json`
- `assets/configs/q2/*.json`

This document explains how those profiles bundle dataset metadata, preprocessing pipelines, and model selections so the notebooks can iterate without extra Python code.

---

## Common Profile Fields

| Field | Type | Purpose |
| --- | --- | --- |
| `dataset` | `string` | Name of the registered dataset loader (`"stock_market"` or `"credit_card_fraud"`). |
| `dataset_options` | `object` | Loader arguments (tickers, parse_dates, row limits, etc.). |
| `split` | `object` | Train/test split parameters (`{"method": "time", "test_size": 0.2}` by default). |
| `target_column` | `string` | Name of the label column after preprocessing (e.g. `"target"` or `"Class"`). |
| `drop_columns` | `array` | Columns to remove before modelling (timestamps, tickers, etc.). |
| `models` | `array` | Traditional model keys from `assets/config.json["models"]`. |
| `neural_models` | `array` | Optional PyTorch model keys from `assets/config.json["deep_models"]`. |
| `metrics` | `array` | Metrics computed in the notebooks (currently `accuracy`, `f1`). |
| `transforms` | `object` | Named transform configs (see [`preprocessing_config.md`](./preprocessing_config.md)). |
| `pipeline` | `array` | Ordered list of transform names to execute. |

### Flow inside the notebooks

1. `load_pipeline_config(profile_path)` reads the JSON.
2. `build_pipeline_from_config(cfg)` returns `(DFXPipeline, metadata)`, where `metadata` mirrors `target_column`, `drop_columns`, etc.
3. The notebook:
   - Loads raw data via the dataset key plus options,
   - Applies `pipeline.apply_global`,
   - Splits the frame according to `split`,
   - Runs `fit_transform` / `transform`,
   - Evaluates the configured models.

This means adding an experiment is as simple as dropping a new JSON file in the right directory.

---

## Dataset Options

Options are forwarded untouched to the dataset loader:

- **Stock market** (`StockMarketDataset`):
  - `tickers`, `date_column`, `ticker_column`, `parse_dates`, `limit_per_ticker`.
- **Credit card fraud** (`CreditCardDataset`):
  - `parse_dates`, `limit_rows` (useful for quick debugging).

Unsupported keys are ignored, so you can extend the loaders without breaking existing configs.

---

## Example Profiles

### Q1 – Baseline (`assets/configs/q1/P0_baseline.json`)
```json
{
  "dataset": "stock_market",
  "dataset_options": {
    "tickers": ["aapl.us", "msft.us"],
    "limit_per_ticker": 2000,
    "parse_dates": true
  },
  "split": { "method": "time", "test_size": 0.2 },
  "target_column": "target",
  "drop_columns": ["Date", "Ticker"],
  "models": ["logistic_regression", "svm", "random_forest"],
  "metrics": ["accuracy", "f1"],
  "transforms": {
    "sort":   { "type": "sorting", "stage": "global", "params": { "order": ["Ticker", "Date"] } },
    "target": { "type": "target",  "stage": "global", "params": { "target_type": "direction", "groupby": ["Ticker"] } },
    "scale":  { "type": "scaling", "stage": "train_test", "params": { "strategy": "standard" } }
  },
  "pipeline": ["sort", "target", "scale"]
}
```

### Q2 – Neural-Friendly Variant (`assets/configs/q2/P1_robust_with_winsorize.json`)
```json
{
  "dataset": "credit_card_fraud",
  "dataset_options": { "limit_rows": 10000 },
  "split": { "method": "time", "test_size": 0.2 },
  "target_column": "Class",
  "models": ["logistic_regression", "gradient_boosting"],
  "neural_models": ["mlp", "residual_mlp"],
  "metrics": ["accuracy", "f1"],
  "transforms": {
    "missing": { "type": "missing_values", "params": { "preset": "credit_card_fraud" } },
    "clip":    { "type": "outlier_clip",    "params": { "method": "winsorize", "lower": 0.01, "upper": 0.99 } },
    "scale":   { "type": "scaling",         "params": { "strategy": "robust" } }
  },
  "pipeline": ["missing", "clip", "scale"]
}
```

---

## Tips for Creating New Profiles

1. **Start from an existing JSON**, duplicate it, and tweak transform parameters.
2. **Validate transform names** against `TRANSFORM_REGISTRY` in `data_processings/transforms.py`.
3. **Specify stages explicitly** when transforms should run only on training data (e.g., class balancing).
4. **Record new profiles in version control**; the notebooks automatically pick up any `*.json` in the directory.
5. **Document intent** inside the JSON using short comments with the `//` JSONC style if needed (the loader tolerates them), or track rationale in `codex.md`.

By standardising on profile files, experiment reproducibility improves and collaborators can reuse pipelines without editing Python.
