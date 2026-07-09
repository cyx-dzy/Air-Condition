from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import ModelConfig


class AttentionForecastNet(nn.Module):
    def __init__(self, feature_count: int, horizon: int, config: ModelConfig):
        super().__init__()
        self.lstm_in = nn.LSTM(feature_count, config.hidden_size, batch_first=True)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.lstm_out = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm_in(x)
        x, _ = self.attention(x, x, x)
        x, _ = self.lstm_out(x)
        return self.head(self.dropout(x[:, -1, :]))


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=(0, 1), keepdims=True)
    std = x.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std, mean, std


def train_forecaster(
    features: np.ndarray,
    target: np.ndarray,
    config: ModelConfig,
) -> tuple[AttentionForecastNet, dict[str, np.ndarray]]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    x = np.transpose(features, (0, 2, 1)).astype(np.float32)
    y = target.astype(np.float32)
    x_scaled, x_mean, x_std = _standardize(x)
    y_mean = y.mean(0, keepdims=True)
    y_std = np.where(y.std(0, keepdims=True) < 1e-6, 1.0, y.std(0, keepdims=True))
    y_scaled = (y - y_mean) / y_std

    model = AttentionForecastNet(feature_count=x.shape[-1], horizon=y.shape[-1], config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    x_tensor = torch.from_numpy(x_scaled)
    y_tensor = torch.from_numpy(y_scaled)

    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad()
        loss = criterion(model(x_tensor), y_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    scaler = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    return model, scaler


def predict(model: AttentionForecastNet, scaler: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    x = np.transpose(features[None, :, :], (0, 2, 1)).astype(np.float32)
    x_scaled = (x - scaler["x_mean"]) / scaler["x_std"]
    model.eval()
    with torch.no_grad():
        y_scaled = model(torch.from_numpy(x_scaled)).numpy()
    return (y_scaled * scaler["y_std"] + scaler["y_mean"])[0]


def training_predictions(
    model: AttentionForecastNet,
    scaler: dict[str, np.ndarray],
    features: np.ndarray,
) -> np.ndarray:
    x = np.transpose(features, (0, 2, 1)).astype(np.float32)
    x_scaled = (x - scaler["x_mean"]) / scaler["x_std"]
    model.eval()
    with torch.no_grad():
        y_scaled = model(torch.from_numpy(x_scaled)).numpy()
    return y_scaled * scaler["y_std"] + scaler["y_mean"]

