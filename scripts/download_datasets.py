"""Download datasets listed in ``assets/dataset_link.json`` into ``assets/datasets``.

The script expects a Kaggle API token to be available (``~/.kaggle/kaggle.json``)
and will download every dataset declared in the JSON file unless specific dataset
names are provided as command line arguments.
"""

from pathlib import Path
from typing import List
import json
import sys

import kaggle


MAIN_DIR = Path(__file__).resolve().parents[1]
DATASETS_ROOT = MAIN_DIR / "assets" / "datasets"
CONFIG_PATH = MAIN_DIR / "assets" / "dataset_link.json"


def _load_dataset_config(config_path: Path) -> dict:
    with config_path.open() as config_file:
        raw_config = json.load(config_file)

    if isinstance(raw_config, dict):
        datasets = raw_config.get("datasets")
    elif isinstance(raw_config, list):
        datasets = raw_config
    else:
        raise ValueError("Dataset configuration must be a list or contain a 'datasets' list.")

    if not datasets:
        raise ValueError("No datasets configured in dataset_link.json.")

    normalised = {}
    for entry in datasets:
        if not isinstance(entry, dict):
            raise ValueError("Each dataset entry must be a JSON object.")

        name = entry.get("name")
        ref = entry.get("kaggle")
        if not name or not ref:
            raise ValueError("Each dataset must define 'name' and 'kaggle' keys.")

        target_dir = entry.get("target_dir") or name
        normalised[name] = {"ref": ref, "target_dir": target_dir}

    return normalised


def download_dataset(name: str, ref: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    print(f"Downloading '{name}' from Kaggle dataset '{ref}' → {destination}")
    kaggle.api.dataset_download_files(ref, path=str(destination), unzip=True, quiet=False)
    print(f"Finished downloading '{name}'.")


def main(args: List[str]) -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Dataset configuration file not found: {CONFIG_PATH}")

    datasets = _load_dataset_config(CONFIG_PATH)

    selected = set(args) if args else set(datasets.keys())
    unknown = selected - datasets.keys()
    if unknown:
        raise SystemExit(f"Unknown dataset name(s) requested: {', '.join(sorted(unknown))}")

    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

    kaggle.api.authenticate()

    for dataset_name in sorted(selected):
        meta = datasets[dataset_name]
        download_dataset(dataset_name, meta["ref"], DATASETS_ROOT / meta["target_dir"])


if __name__ == "__main__":
    main(sys.argv[1:])
