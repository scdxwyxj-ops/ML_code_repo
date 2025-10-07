"""Neural network models and training utilities for tabular datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int], dropout: float, num_classes: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_dim = hidden
        layers.append(nn.Linear(last_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TabularLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
        self.input_dim = input_dim

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        seq = x.view(batch_size, self.input_dim, 1)
        output, _ = self.lstm(seq)
        pooled = output[:, -1, :]
        return self.classifier(pooled)


class TabularTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self.input_dim = input_dim

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        seq = x.view(batch_size, self.input_dim, 1)
        seq = self.input_proj(seq)  # (batch, seq_len, d_model)
        encoded = self.encoder(seq)
        pooled = encoded.mean(dim=1)
        return self.output_head(pooled)


def build_neural_model(
    key: str,
    input_dim: int,
    config: Mapping[str, Mapping[str, object]],
    num_classes: int = 2,
) -> Tuple[nn.Module, TrainingConfig]:
    if key not in config:
        raise KeyError(f"Unknown neural model '{key}'")

    model_cfg = config[key]
    params = model_cfg.get("params", {})
    training_kwargs = model_cfg.get("training", {})
    training_config = TrainingConfig(
        epochs=int(training_kwargs.get("epochs", 20)),
        batch_size=int(training_kwargs.get("batch_size", 256)),
        learning_rate=float(training_kwargs.get("learning_rate", 1e-3)),
    )

    if model_cfg.get("class") == "TabularMLP":
        hidden_dims = params.get("hidden_dims", [128, 64])
        dropout = float(params.get("dropout", 0.2))
        model = TabularMLP(input_dim, hidden_dims, dropout, num_classes)
    elif model_cfg.get("class") == "TabularLSTM":
        hidden_size = int(params.get("hidden_size", 64))
        num_layers = int(params.get("num_layers", 2))
        dropout = float(params.get("dropout", 0.2))
        model = TabularLSTM(input_dim, hidden_size, num_layers, dropout, num_classes)
    elif model_cfg.get("class") == "TabularTransformer":
        d_model = int(params.get("d_model", 64))
        nhead = int(params.get("nhead", 8))
        num_layers = int(params.get("num_layers", 2))
        dim_feedforward = int(params.get("dim_feedforward", 128))
        dropout = float(params.get("dropout", 0.1))
        model = TabularTransformer(
            input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unsupported neural model class: {model_cfg.get('class')}")

    return model, training_config


def build_dataloader(
    features: Tensor,
    labels: Tensor,
    batch_size: int,
    shuffle: bool = False,
    pin_memory: bool | None = None,
) -> DataLoader:
    dataset = TensorDataset(features, labels)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)


def train_neural_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    class_weights: Tensor | None = None,
    device: torch.device | None = None,
    progress_label: str | None = None,
    save_path: Path | None = None,
    metadata: Mapping[str, object] | None = None,
) -> list[Dict[str, float]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_iter = range(epochs)
    if progress_label:
        epoch_iter = tqdm(epoch_iter, desc=progress_label, leave=False)

    history: list[Dict[str, float]] = []
    for epoch in epoch_iter:
        model.train()
        train_loss = 0.0
        for batch_features, batch_labels in train_loader:
            features = batch_features.to(device, non_blocking=True)
            labels = batch_labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(features)
                loss = criterion(logits, labels)
                val_loss += loss.item() * features.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / max(total, 1)
        history.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(avg_train_loss),
                "val_loss": float(avg_val_loss),
                "val_accuracy": float(val_accuracy),
            }
        )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": model.state_dict(),
            "metadata": dict(metadata or {}),
        }
        torch.save(payload, save_path)

    return history


def predict_proba(model: nn.Module, loader: DataLoader) -> Tensor:
    device = next(model.parameters()).device
    model.eval()
    outputs: list[Tensor] = []
    with torch.no_grad():
        for features, _ in loader:
            features = features.to(device, non_blocking=True)
            logits = model(features)
            outputs.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(outputs, dim=0)


__all__ = [
    "TabularMLP",
    "TabularLSTM",
    "TabularTransformer",
    "TrainingConfig",
    "build_dataloader",
    "build_neural_model",
    "predict_proba",
    "train_neural_model",
]
