# Models Configuration (`models`, `deep_models`)

This document explains how models are declared inside `assets/config.json` and how they are consumed by the codebase.

---

## Traditional Models (`models`)

Entries under `models` define scikit-learn compatible estimators instantiated via `models/registry.py` (accessed with `from models import build_model`).

### Schema

```json
"model_key": {
  "module": "python.import.path",
  "class": "EstimatorClassName",
  "params": { ... optional keyword arguments ... }
}
```

- **`module`** – Python import path for the estimator (e.g. `sklearn.linear_model`).
- **`class`** – Class name inside the module (e.g. `LogisticRegression`).
- **`params`** – Default keyword arguments passed to the constructor.

These models are typically simple classifiers (logistic regression, naive Bayes, decision tree, SVM, random forest, gradient boosting). The Q1 and Q2 notebooks iterate through the model keys specified in their respective experiment blocks.

---

## Neural Models (`deep_models`)

Neural definitions are consumed by `models/neural.py`, which builds the PyTorch module and returns both the model and training configuration.

### Schema

```json
"model_key": {
  "module": "models.neural",
  "class": "TabularMLP|ResidualMLP|TabularTransformer",
  "params": { ... architecture hyperparameters ... },
  "training": {
    "epochs": <int>,
    "batch_size": <int>,
    "learning_rate": <float>
  }
}
```

Supported classes:

| Class | Parameters | Description |
| --- | --- | --- |
| `TabularMLP` | `hidden_dims` (list), `dropout` | Fully connected MLP with ReLU + dropout. |
| `ResidualMLP` | `hidden_dim`, `num_blocks`, `dropout` | Residual MLP with layer-norm skip connections. |
| `TabularTransformer` | `d_model`, `nhead`, `num_layers`, `dim_feedforward`, `dropout` | Transformer encoder for tabular sequences (batch-first). |

The training block defines defaults for epochs, batch size, learning rate, and optional gradient clipping (`grad_clip`). The Q2 notebook reads these values and stores the resulting checkpoints/history under `assets/snapshots`.

---

## Customising Models

1. **Add a new traditional model**:
   ```json
   "extra_svm": {
     "module": "sklearn.svm",
     "class": "LinearSVC",
     "params": {
       "C": 0.5,
       "class_weight": "balanced"
     }
   }
   ```
   Then include `"extra_svm"` in the experiment’s `models` list.

2. **Add a new neural architecture**: Extend `models/neural.py` with the implementation, register it under `deep_models`, and add the key to the experiment’s `neural_models` list.

3. **Override parameters at runtime**: For traditional models, use `build_model("model_key", overrides={"param": value})`. Neural overrides are typically handled by tweaking the configuration directly.
