# Financial Market & Fraud ML Toolkit

Repository for COMP90049 (Introduction to Machine Learning) Assignment 2, Group G03.  
It provides a configuration-driven pipeline for:
- **Q1** &ndash; Stock price movement ablation studies across feature engineering variants.
- **Q2** &ndash; Credit-card fraud detection comparing classical ML and neural baselines.
- **Q3** &ndash; Temporal stability comparison of different ML-based credit risk models.
- **Q4** &ndash; Stock-market realized volatility forecasting and ML-driven portfolio backtesting.
- **Q5** &ndash; Portfolio construction benchmarking Markowitz optimisation against ML strategies.

All preprocessing, feature engineering, and model choices are declarative; notebooks simply load JSON profiles and execute the shared transform stack. 

---

## Highlights
- **Composable transforms** – `data_processings.transforms.DFXTransform` and `DFXPipeline` enable reusable, stage-aware preprocessing that mirrors PyTorch-style transforms.
- **Config-first experiments** – Each ablation profile lives in `assets/configs/q1|q2/*.json`, defining dataset options, transform order, and model lists without editing Python.
- **Dataset abstractions** – `data_processings.datasets.BaseDataset` is subclassed for stock-market and credit-card loaders with consistent option handling.
- **Progress-aware experiment scripts** – `scripts/q1.ipynb` and `scripts/q2.ipynb` emit clear logs for classical evaluation, neural training, and neural inference; neural checkpoints are cached under `assets/snapshots`. `scripts/q3.ipynb` surfaces temporal stability outputs inline and archives supporting tables in `assets/q3_data`. `scripts/q4.py` automates the volatility forecasting workflow plus portfolio backtesting, exporting artefacts to `results/` and `portfolio_results/`. `scripts/q5.ipynb` walks through Markowitz vs ML portfolio comparisons with narrative commentary.
- **Documentation bundle** – `assets/documentation/` captures environment setup, config schema, and experiment workflows for quick onboarding.

---

## Quick Start
1. **Set up the environment**
   ```bash
   conda env create -f environment.yml
   conda activate ML
   ```
2. **Download datasets** (requires a Kaggle API token in `~/.kaggle/kaggle.json`)
   ```bash
   python scripts/download_datasets.py stock_market credit_card_fraud
   ```
3. **Launch Jupyter from the repository root**
   ```bash
   jupyter lab  # or: jupyter notebook
   ```
4. **Run the notebooks from `scripts/`**
   - `scripts/q1.ipynb` (stock ablations)  
   - `scripts/q2.ipynb` (credit fraud classical + neural)
   - `scripts/q3.ipynb` (temporal stability of credit risk models)
   - `scripts/q5.ipynb` (Markowitz vs ML portfolio construction)

   Execute command-line workflows from the same folder when required:
   - `python q4.py` (volatility forecasting and portfolio backtests)

   Each notebook derives `MAIN_PATH = Path(os.getcwd()).parent` or equivalent, so execute them while your working directory is `scripts/`.

---

## Configuring Experiments
- **Shared presets** – `assets/config.json`
  - `missing_value_presets`: named imputation strategies.
  - `models`: scikit-learn estimators used by both notebooks.
  - `deep_models`: PyTorch architectures plus training defaults for Q2.
- **Per-question profiles** – `assets/configs/q1/*.json`, `assets/configs/q2/*.json`
  - `dataset` + `dataset_options`: forwarded to the dataset classes.
  - `transforms`: ordered, stage-aware transform definitions (see `data_processings/transforms.py`).
  - `pipeline`: execution order for the named transforms.
  - `models`, `neural_models`, `metrics`: control which models/metrics the notebooks run.
  - Metadata such as `target_column`, `drop_columns`, and `split` is surfaced to the notebooks automatically.

Modify or clone a profile to create a new ablation; no Python changes are required as long as the transform parameters are valid.

---

## Documentation
The `assets/documentation/` directory contains:
- `documentation.md` – end-to-end project guide (environment, datasets, workflow, troubleshooting).
- `configs/` – detailed schema references for shared presets, model registries, and pipeline JSON profiles.

For deeper architectural context and recent changes, see `codex.md`, which is updated every refactor.

---

## Repository At A Glance
- `data_processings/` – Transform base classes, registry, dataset loaders, scaling/outlier utilities, and pipeline builder.
- `models/` – Traditional model registry plus PyTorch tabular models.
- `scripts/` – Dataset downloader, experiment notebooks (`q1.ipynb`, `q2.ipynb`, `q3.ipynb`, `q5.ipynb`), and the Q4 volatility pipeline (`q4.py`).
- `assets/`
  - `config.json` – shared presets.
  - `configs/q1|q2` – ablation profiles.
  - `datasets/` – local copies of Kaggle datasets (ignored by Git).
  - `snapshots/` – neural checkpoints and training histories.
  - `q3_data/` – outcomes and details of q3 credit risk models. 

---

## Maintainers
- Zivanka Nafisa Wongkaren (1446841)  
- Kim Donguk (1674775)  
- Zhijing Qiu (1637936)  
- Jun Xu (1550679)

Feel free to open issues or pull requests if you extend the pipelines or add new experiments.
