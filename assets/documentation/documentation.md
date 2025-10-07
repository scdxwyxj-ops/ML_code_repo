# Project Documentation

This guide explains how to work with the ML toolkit developed for COMP90049 Assignment 2 (Group G03). It covers the repository layout, environment setup, dataset handling, configuration system, and experiment execution.

---

## 1. Repository Layout

```
assets/
  config.json           # Shared presets (missing-value schemas, model registries)
  configs/
    q1/*.json           # Stock-market ablation profiles
    q2/*.json           # Credit-fraud ablation profiles
  datasets/             # Kaggle datasets (ignored by Git)
  documentation/        # This documentation bundle
  snapshots/            # Neural checkpoints & training histories
data_processings/
  datasets.py           # BaseDataset + concrete loaders
  transforms.py         # DFXTransform hierarchy & registry
  pipeline_builder.py   # JSON → DFXPipeline converter
  ...                   # missing_values, scaling, rolling features, etc.
models/
  registry.py           # Traditional model factory
  neural.py             # PyTorch tabular models & training helpers
scripts/
  download_datasets.py  # Kaggle downloader
  q1.ipynb              # Stock ablation workflow
  q2.ipynb              # Credit-fraud workflow
codex.md                # Running development notes
README.md               # High-level overview & quick start
```

Key ideas:
- **Config-first design** – Notebook logic is stable; ablations are defined in JSON under `assets/configs/q1|q2`.
- **Composable transforms** – `DFXTransform` subclasses implement fit/transform hooks with stage-awareness (`global`, `train_test`, `train_only`, `test_only`).
- **Dataset abstraction** – Loaders inherit from `BaseDataset`, so new datasets only need loader subclasses plus JSON profiles.

---

## 2. Environment Setup

1. Install Miniconda/Anaconda if required.
2. Create and activate the project environment:
   ```bash
   conda env create -f environment.yml
   conda activate ML
   ```
3. (Optional) Add extra packages as needed. Update `environment.yml` if the dependency should be shared.

The environment ships with Python 3.12, pandas, scikit-learn, PyTorch (CPU), tqdm, Jupyter, and the Kaggle CLI.

---

## 3. Dataset Management

1. Place your Kaggle API token at `~/.kaggle/kaggle.json` with permission `600`.
2. Download datasets from the repository root:
   ```bash
   python scripts/download_datasets.py stock_market credit_card_fraud
   ```
3. Data lands in:
   - `assets/datasets/stock_market/**` (stock/ETF CSVs)
   - `assets/datasets/credit_card_fraud/creditcard.csv`

The loaders accept options such as `tickers`, `limit_per_ticker`, or `limit_rows` so experiments can trim dataset size without editing code.

---

## 4. Configuration System

### 4.1 Shared Presets (`assets/config.json`)

| Key | Purpose |
| --- | --- |
| `missing_value_presets` | Named imputation strategies consumed by the `missing_values` transform. |
| `models` | scikit-learn estimators, built by `models.build_model(model_key)`. |
| `deep_models` | PyTorch architectures and training defaults used by Q2. |

These presets are read once and cached via `data_processings.config.load_config`.

### 4.2 Ablation Profiles (`assets/configs/q1|q2/*.json`)

Each profile is self-contained:
- `dataset` / `dataset_options` – forwarded to the dataset loader.
- `split` – currently `{"method": "time", "test_size": 0.2}`.
- `target_column`, `drop_columns` – surfaced to the notebooks for label extraction.
- `models`, `neural_models`, `metrics` – determine which estimators run.
- `transforms` – mapping of named transform configs (`type`, `params`, optional `stage`, `enabled`).
- `pipeline` – ordered list of transform names to execute.

Profiles are parsed with:
```python
cfg = load_pipeline_config(path)
pipeline, metadata = build_pipeline_from_config(cfg)
```
The resulting `metadata` dictionary carries the target/drop metadata into the notebook workflow.

### 4.3 Transform Stages

| Stage | Runs During… | Typical Use |
| --- | --- | --- |
| `global` | `pipeline.apply_global(df)` | Sorting, dataset-wide feature engineering, target creation |
| `train_test` | `fit_transform(train)` + `transform(test)` | Scaling, outlier clipping, log transforms |
| `train_only` | `fit_transform(train)` only | SMOTE/oversampling, augmentation |
| `test_only` | `transform(test)` only | Evaluation-specific tweaks |

Stage aliases (`pre`, `post`, `both`, `train`, `test`) map onto these canonical stages.

---

## 5. Running Experiments

### 5.1 Notebook Logistics

Both notebooks live in `scripts/` and assume `os.getcwd()` equals that directory.
Each notebook begins with:
```python
MAIN_PATH = Path(os.getcwd()).parent
sys.path.append(str(MAIN_PATH))
```
Run them by navigating to `scripts/` inside Jupyter before executing any cells.

### 5.2 `scripts/q1.ipynb` – Stock Ablations

Workflow:
1. Discover all profiles under `assets/configs/q1`.
2. Load raw data via `StockMarketDataset`.
3. Apply global transforms, perform a time-based split, and run train/test-stage transforms.
4. Evaluate the configured classical models (`accuracy`, `f1` by default).
5. Produce a tidy `results_df`.

Helpful behaviours:
- Sanitises NaNs/inf values before modelling.
- Aligns feature columns between train/test splits automatically.
- Emits progress logs per profile/model.

### 5.3 `scripts/q2.ipynb` – Credit-Card Fraud (Classical + Neural)

Workflow:
1. Load credit-card data (optionally limited via `dataset_options.limit_rows`).
2. Build `DFXPipeline` per profile and prepare train/test matrices.
3. Evaluate classical models (progress printed).
4. Train neural models when snapshots are absent; training metadata records sample counts and metrics, saved to `assets/snapshots`.
5. Reload snapshots for evaluation; the summary dataframe contains both classical and neural results plus train/test sample counts.

Snapshots can be deleted to retrain from scratch; the notebook will recreate them on demand.

---

## 6. Extending the Toolkit

1. **Add a new transform** – Subclass `DFXTransform`, register it in `TRANSFORM_REGISTRY`, then reference it in a profile JSON.
2. **Add a dataset** – Implement a `BaseDataset` subclass and point `dataset` to its registry name; supply options via `dataset_options`.
3. **Create a new profile** – Copy an existing JSON file, tweak transforms/parameters, and re-run the notebook. The workflow automatically picks up new files.
4. **Integrate a new model** – Register it in `assets/config.json` under `models` or `deep_models`, then list its key in the profile.

---

## 7. Troubleshooting & Tips

- **Module import errors** – Ensure the notebook’s working directory is `scripts/` so `MAIN_PATH` resolves to the repository root.
- **Missing columns after preprocessing** – Check transform stages; target construction must occur in a `global` stage before train/test splits.
- **Neural `train_samples` appearing as `NaN`** – The notebook now records counts in checkpoint metadata; delete stale snapshots if they were created before this update.
- **Performance issues** – Lower `dataset_options.limit_rows` (Q2) or `limit_per_ticker` (Q1) for faster iterative runs.
- **Snapshots** – Stored exclusively in `assets/snapshots`. Remove specific files to retrain, or delete the entire directory to regenerate everything.

---

## 8. Further Reading

- [`configs/config_overview.md`](./configs/config_overview.md) – High-level schema of `assets/config.json`.
- [`configs/preprocessing_config.md`](./configs/preprocessing_config.md) – Transform profile format and registry reference.
- [`configs/models_config.md`](./configs/models_config.md) – Traditional and neural model configuration examples.
- [`configs/experiments_config.md`](./configs/experiments_config.md) – How profiles bundle datasets, transforms, and model lists.
- `codex.md` – Development log capturing design decisions, caveats, and future ideas.

---

With the environment prepared and configurations understood, you can iterate rapidly: tweak a JSON profile, rerun the notebook, and compare results without editing Python modules. Happy experimenting!
