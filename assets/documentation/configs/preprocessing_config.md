# Preprocessing Pipelines (`assets/configs/<question>/*.json`)

Each ablation profile is described by a standalone JSON file under `assets/configs/q1` or `assets/configs/q2`. The format is intentionally verbose: every transform declares its type, parameters, and execution stage so the pipeline can be rebuilt without touching Python code.

## File Structure

```jsonc
{
  "dataset": "stock_market",
  "dataset_options": { "tickers": ["aapl.us", "msft.us"], "parse_dates": true },
  "split": { "method": "time", "test_size": 0.2 },
  "target_column": "target",
  "drop_columns": ["Date", "Ticker"],
  "models": ["logistic_regression", "svm"],
  "metrics": ["accuracy", "f1"],
  "transforms": {
    "sort": { "type": "sorting", "stage": "global", "params": { "order": ["Ticker", "Date"] } },
    "missing": { "type": "missing_values", "stage": "global", "params": { "preset": "stock_market" } },
    "...": { "...": "..." }
  },
  "pipeline": ["sort", "missing", "..."]
}
```

- **dataset / dataset_options** feed directly into the dataset loader. Options are passed unmodified to `StockMarketDataset.load` or `CreditCardDataset.load`.
- **split** controls how `split_time_series_frame` separates train/test data (currently time-based splits only).
- **target_column** and optional **drop_columns** are surfaced as metadata so notebooks can pop the label and drop identifiers prior to modelling.
- **models** and **metrics** declare which classical estimators to evaluate per profile.
- **transforms** is a named mapping. Each entry must include a `type` (resolved through `data_processings.transforms.TRANSFORM_REGISTRY`). Optional fields:
  - `params`: keyword arguments forwarded to the transform constructor (e.g., scaler strategy, rolling specs).
  - `stage`: controls when the transform runs (see below).
  - `enabled`: `false` removes the transform from the pipeline without deleting its configuration.
  - `apply_to`: legacy synonym for stage (`"train"` → `train_only`, `"both"` → `train_test`).
- **pipeline** defines the execution order. Only transforms listed here will run.

## Stage Semantics

| Stage value        | Runs during…                     | Typical usage                               |
|--------------------|----------------------------------|---------------------------------------------|
| `global`           | `pipeline.apply_global(df)`      | Sorting, dataset-wide feature engineering, target creation |
| `train_test` (default) | `fit_transform(train_df)` and `transform(test_df)` | Scaling, winsorisation, log transforms |
| `train_only`       | `fit_transform(train_df)` only   | Oversampling/undersampling, train-specific augmentation |
| `test_only`        | `transform(test_df)` only        | Inference-time tweaks (rare)                |
| `all`              | Every stage                      | Diagnostics or bookkeeping transforms       |


The DFX base class enforces deterministic ordering: global transforms run once, train/test stages share fitted state, and disabled transforms are skipped entirely.

## Available Transform Types

| Type (`transforms.<name>.type`) | Description |
|---------------------------------|-------------|
| `sorting` | Sort rows by the provided column order. |
| `missing_values` | Impute using presets from `assets/config.json` (`stage` defaults to `train_test`; set `stage: "global"` to reuse whole-dataset stats). |
| `constant_fill` | Column-wise fill values, optionally emitting `<col>_was_missing`. |
| `log1p` / `signed_log1p` | Apply log transforms to selected numeric columns. |
| `returns` | Derive percentage change features with optional per-ticker grouping. |
| `rolling` | Rolling-window aggregations delegated to `apply_rolling_features`. |
| `feature_set` | Batch feature engineering steps (technical indicators, sentiment proxies, macro proxies, etc.). |
| `target` | Build modelling targets (directional price moves or column copies). Runs with `stage: "global"` by default. |
| `drop_columns` | Remove unneeded columns before modelling. |
| `class_balance` | Oversample / undersample training data (`stage: "train_only"`). |
| `outlier_clip` | Winsorise or clip outliers; parameters map to `fit_outlier_clip_model`. |
| `scaling` | Standard/robust/min–max scaling with optional group-by columns. |
| `column_map` | Apply arbitrary vectorised functions to selected columns. |

Refer to `data_processings/transforms.py` for constructor signatures and `TRANSFORM_REGISTRY` mappings.

## Example

`assets/configs/q1/P0_baseline.json` combines global feature engineering with post-split scaling:

```json
"transforms": {
  "sort":   { "type": "sorting",        "stage": "global",     "params": { "order": ["Ticker", "Date"] } },
  "missing":{ "type": "missing_values", "stage": "global",     "params": { "preset": "stock_market" } },
  "features_all": {
    "type": "feature_set",
    "stage": "global",
    "params": { "steps": [{ "type": "technical_indicators", ... }, { "type": "sentiment_proxy", ... }] }
  },
  "rolling":{ "type": "rolling",        "stage": "global",     "params": { "specs": [{ "columns": ["Close"], "window": 5, ... }] } },
  "target": { "type": "target",         "stage": "global",     "params": { "target_type": "direction", "groupby": ["Ticker"] } },
  "scale":  { "type": "scaling",        "stage": "train_test", "params": { "strategy": "standard", "columns": ["Open", "High", "Low", "Close", "Volume"], "groupby": ["Ticker"] } }
},
"pipeline": ["sort", "missing", "features_all", "rolling", "target", "scale"]
```

The notebooks load these configs with:

```python
cfg = load_pipeline_config(path)
pipeline, metadata = build_pipeline_from_config(cfg)
processed_global = pipeline.apply_global(raw_df)
train_df = pipeline.fit_transform(train_split)
test_df = pipeline.transform(test_split)
```

This keeps the Python surface area small—new experiments are defined entirely in JSON.
