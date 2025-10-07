# Experiments Configuration (`experiments`)

The `experiments` section of `assets/config.json` brings preprocessing, models, and datasets together. Each key corresponds to a notebook or script entry point (e.g. `q1_stock_movement`, `q2_credit_fraud`).

## Common Fields

| Field | Type | Description |
| --- | --- | --- |
| `dataset` | `string` | Name of the dataset in the `preprocessing` block (e.g. `stock_market`). |
| `dataset_options` | `object` | Loader-specific options passed to the dataset helper (tickers, parse_dates, row limits, etc.). |
| `split` | `object` | Evaluation split settings (currently `method: "time"` with `test_size`). |
| `models` | `array` | Keys from the traditional `models` registry to evaluate. |
| `neural_models` | `array` | Keys from `deep_models` (optional; used in Q2). |
| `metrics` | `array` | Metric names computed in the notebook (supported: `accuracy`, `f1`). |
| `ablation_axis` | `string` | Either `"features"` or `"preprocessing"`; determines how the ablation loop iterates. |
| `ablation_sets_key` | `string` | Reference into the dataset’s preprocessing entry (`ablation_feature_sets` or `ablation_preprocessing_sets`). |
| `feature_sets_fixed` | `array` | When `ablation_axis == "preprocessing"`, the fixed feature sets to apply. |

### `dataset_options`

These are forwarded to dataset loaders (`load_stock_market_data`, `load_credit_card_data`). Examples:

- `tickers`: List of symbols to load (stock market).
- `date_column`, `ticker_column`, `parse_dates`: CSV parsing controls.
- `limit_per_ticker` / `limit_rows`: Trim dataset size for quick iterations.

### `split`

Only time-based splits are currently supported:

```json
"split": {
  "method": "time",
  "test_size": 0.2
}
```

Notebooks convert this into a chronological train/test split. Training data may then be balanced (if `class_balance` is configured) before scaling/outlier transforms.

### `ablation_axis` and `ablation_sets_key`

- `"features"` – Iterate over combinations defined in `preprocessing.<dataset>.ablation_feature_sets`.
- `"preprocessing"` – Iterate over profile names listed in `preprocessing.<dataset>.ablation_preprocessing_sets` while using `feature_sets_fixed`.

### `neural_models`

When present, the experiment is expected to train or evaluate neural architectures defined in `deep_models`. The Q2 notebook:

1. Calls `train_neural_profiles` to generate or reuse snapshots.
2. Reloads those snapshots for inference.

## Example: `q2_credit_fraud`

```json
"q2_credit_fraud": {
  "dataset": "credit_card_fraud",
  "dataset_options": {
    "parse_dates": false,
    "limit_rows": 200000
  },
  "split": { "method": "time", "test_size": 0.2 },
  "models": [
    "logistic_regression",
    "naive_bayes",
    "decision_tree",
    "svm",
    "random_forest",
    "gradient_boosting"
  ],
  "neural_models": ["mlp", "lstm", "transformer"],
  "metrics": ["accuracy", "f1"],
  "ablation_axis": "preprocessing",
  "ablation_sets_key": "ablation_preprocessing_sets",
  "feature_sets_fixed": ["baseline"]
}
```

This configuration instructs the notebook to:

1. Load the credit-card fraud data.
2. Iterate over preprocessing profiles (`P0`–`P3`) defined in the dataset entry.
3. Apply classical models plus neural baselines (loading snapshots if available).
4. Report accuracy/F1 for each profile-model combination.
