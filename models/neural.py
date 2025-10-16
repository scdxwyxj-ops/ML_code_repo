"""Neural network models and training utilities for tabular datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from torchsummary import summary
from sklearn.model_selection import train_test_split
import torch.optim as optim
import contextlib
import os


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    grad_clip: float | None = None


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


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self.block(x))


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        dropout: float,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


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
        self.input_norm = nn.LayerNorm(d_model)
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
        seq = self.input_norm(seq)
        encoded = self.encoder(seq)
        pooled = encoded.mean(dim=1)
        return self.output_head(pooled)

class ClassifierNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_layer_neurons, output_size, dropout_perc=0.2):
        super(ClassifierNeuralNet, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_layer_neurons)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_perc)

        self.layer2 = nn.Linear(hidden_layer_neurons, hidden_layer_neurons)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_perc)

        self.layer3 = nn.Linear(hidden_layer_neurons, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.layer3(x)

        return x
    
class BinaryClassifier():

    def __init__(self, input_size):
        self.input_size = input_size
        self.hidden_layer_neurons = 128
        self.output_size = 2 # Binary
        self.dropout_percent = 0.3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = ClassifierNeuralNet(self.input_size, self.hidden_layer_neurons, self.output_size, self.dropout_percent).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005, amsgrad=True)

    def fit(self, X_train, y_train, epochs=30, val_split=0.2, batch_size=128):
        patience = 10
        best_valid_loss = 100000.0
        batch_loss = 0.0
        ctr = 0

        if val_split > 0:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train, y_train, test_size=val_split, random_state=10, shuffle=True
            )
        else:
            X_valid, y_valid = None, None
        
        train_dataset = TensorDataset(X_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        valid_dataset = TensorDataset(X_valid, y_valid)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):

            for X_train_batch, y_train_batch in train_dataloader:
                self.model.train()
                self.optimizer.zero_grad()
                outputs = self.model(X_train)
                loss = self.criterion(outputs, y_train)
                loss.backward()
                self.optimizer.step()
                batch_loss += loss.item() * X_train_batch.size(0)
            
            training_loss = batch_loss / len(train_dataloader)

            if X_valid is not None:
                self.model.eval()
                valid_loss = 0.0

                with torch.no_grad():
                    for X_valid_batch, y_valid_batch in valid_dataloader:
                        valid_outputs = self.model(X_valid_batch)
                        val_loss = self.criterion(valid_outputs, y_valid_batch)
                        valid_loss += val_loss.item() * X_valid_batch.size(0)
                
                valid_loss = valid_loss / len(valid_dataloader.dataset)    
            else:
                valid_loss = None

            epoch_name = 1+epoch

            if (epoch_name) % 5 == 0:
                if valid_loss is None:
                    print(f'Epoch [{epoch_name}/{epochs}], Training Loss: {training_loss:.4f}')
                else:
                    print(f'Epoch [{epoch_name}/{epochs}], Training Loss: {training_loss:.4f}, Validation Loss: {valid_loss:.4f}')
            
            if valid_loss.item() < best_valid_loss:  
                best_valid_loss = valid_loss.item()
                ctr = 0      
                best_model_state = self.model
            else: 
                ctr += 1
                if ctr >= patience:
                    print("Model Has Not Improved! Training Stopped Early!")
                    break

        self.model = best_model_state

    def predict(self, X_test):

        self.model.eval()

        with torch.no_grad():
            preds = self.model(X_test)
            preds = torch.argmax(preds, dim=1)

        preds = preds.numpy()

        return preds
    
    def print_model_summary(self, folder_path, file_name):
        os.makedirs(folder_path, exist_ok=True)
        file_name = f"{file_name}.txt"
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, "w") as txt_file:
            with contextlib.redirect_stdout(txt_file):
                summary(self.model, input_size=(self.input_size,))

        print(f"Neural Network Details Saved in {file_path}!")

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
        grad_clip=float(training_kwargs["grad_clip"]) if training_kwargs.get("grad_clip") is not None else None,
    )

    if model_cfg.get("class") == "TabularMLP":
        hidden_dims = params.get("hidden_dims", [128, 64])
        dropout = float(params.get("dropout", 0.2))
        model = TabularMLP(input_dim, hidden_dims, dropout, num_classes)
    elif model_cfg.get("class") == "ResidualMLP":
        hidden_dim = int(params.get("hidden_dim", 256))
        num_blocks = int(params.get("num_blocks", 3))
        dropout = float(params.get("dropout", 0.2))
        model = ResidualMLP(input_dim, hidden_dim, num_blocks, dropout, num_classes)
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
    grad_clip: float | None = None,
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
            if grad_clip is not None:
                clip_grad_norm_(model.parameters(), grad_clip)
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
