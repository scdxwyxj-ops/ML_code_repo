# Preprocessing Configuration (`preprocessing`)

The `preprocessing` section of `assets/config.json` defines how raw datasets are cleaned, transformed, and enriched before modelling. Each dataset key (e.g. `stock_market`, `credit_card_fraud`) contains a base template which can be further customised through **profiles**.

## Common Fields

| Field | Type | Description |
| --- | --- | --- |
| `missing_values_preset` | `string` | Name of the preset declared in `missing_value_presets`. |
| `scaling` | `object` | Optional scaler configuration (`strategy`, `columns`, `groupby`, `feature_range`). |
| `order_columns` | `array` | Column order used before imputations (defaults to `["Ticker", "Date"]` where applicable). |
| `rolling_features` | `array` | Rolling-window feature specs (window size, aggregations, groupby). |
| `transforms` | `array` | Lightweight column transforms (`log1p_safe`, `signed_log1p`, `returns`). |
| `feature_sets` | `object` | Named feature-engineering pipelines (see below). |
| `target` | `object` | Target definition (directional or column-based). |
| `feature_selection` | `object` | Columns to exclude from feature matrices. |
| `class_balance` | `object` | Optional training-set balancing (`method`: `oversample`/`undersample`, `random_state`, `target_column`). |
| `profiles` | `object` | Named overrides that adjust any of the fields above. |
| `ablation_preprocessing_sets` | `array` | Ordered list of profile names used when `ablation_axis` is `preprocessing`. |

### Feature Sets

Each entry combines a `steps` array with one or more of the supported feature-engineering operations:

- `technical_indicators` – Adds SMA, volatility, momentum, RSI, and volume statistics. Supports per-group rolling via `groupby`.
- `sentiment_proxy` – Builds sentiment-like signals from prior returns (rolling mean/std/EMA) and shifts them forward to avoid leakage.
- `macro_proxy` – Aggregates per-date statistics across tickers/features and shifts them forward to avoid leakage.
- `external_join` – Merges an external CSV (path, join keys, join type) and optionally fills numeric missing values.

Feature sets can be combined during experiments (e.g. `["technical", "sentiment"]`).

### Profiles

Profiles allow you to alter combinations of settings without changing the base template—for example:

- Switch imputation strategies (`missing_override`).
- Apply different scalers (`scaling.strategy`).
- Introduce rolling windows with alternative sizes.
- Enable outlier clipping via `outlier_clip` (winsorization).
- Toggle balancing strategies with `class_balance`.

During experiment runs, the notebook fetches the selected profile, deep merges it with the base entry, and records which profile produced the results.

## Example: `credit_card_fraud`

```json
"credit_card_fraud": {
  "missing_values_preset": "credit_card_fraud",
  "scaling": {
    "strategy": "standard",
    "columns": ["Time", "Amount", "V1", "...", "V28"]
  },
  "order_columns": ["Time"],
  "feature_sets": { "baseline": { "steps": [] } },
  "target": { "type": "column", "column": "Class" },
  "feature_selection": { "exclude": ["Class"] },
  "class_balance": {
    "method": "oversample",
    "random_state": 42,
    "target_column": "Class"
  },
  "profiles": {
    "P0_baseline": {},
    "P1_robust_with_winsorize": { "...": "..." },
    "P2_minmax_global": { "...": "..." },
    "P3_log_amount_no_scaling": { "...": "..." }
  },
  "ablation_preprocessing_sets": [
    "P0_baseline",
    "P1_robust_with_winsorize",
    "P2_minmax_global",
    "P3_log_amount_no_scaling"
  ]
}
```

Refer to the stock market entry for examples involving rolling features, sentiment/macro proxies, and profile-driven transforms.
