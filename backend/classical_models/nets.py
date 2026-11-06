"""PyTorch architectures. Defined at module level so saved models can be unpickled from Drive."""

from __future__ import annotations

import torch
from torch import nn


class RecurrentRegressor(nn.Module):
    def __init__(self, n_features: int, cell: str = "lstm", hidden: int = 64):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(n_features, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


class CnnLstmRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(n_features, hidden, 5, padding=2), nn.ReLU(),
                                  nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU())
        self.rnn = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.rnn(h)
        return self.head(out[:, -1]).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(self, n_features: int, window: int, d: int = 64):
        super().__init__()
        self.proj = nn.Linear(n_features, d)
        self.pos = nn.Parameter(torch.randn(1, window, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        h = self.enc(self.proj(x) + self.pos[:, :x.shape[1]])
        return self.head(h.mean(1)).squeeze(-1)


class DenseAutoencoder(nn.Module):
    def __init__(self, n_features: int, bottleneck: int = 4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_features, 16), nn.ReLU(), nn.Linear(16, bottleneck))
        self.dec = nn.Sequential(nn.Linear(bottleneck, 16), nn.ReLU(), nn.Linear(16, n_features))

    def forward(self, x):
        return self.dec(self.enc(x))


class LstmAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 16):
        super().__init__()
        self.enc = nn.LSTM(n_features, hidden, batch_first=True)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x):
        _, (h, _) = self.enc(x)
        z = h[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        y, _ = self.dec(z)
        return self.out(y)


def build_rul_net(name: str, n_features: int, window: int) -> nn.Module:
    if name in ("lstm", "gru"):
        return RecurrentRegressor(n_features, cell=name)
    if name == "cnn_lstm":
        return CnnLstmRegressor(n_features)
    if name == "transformer":
        return TransformerRegressor(n_features, window)
    raise ValueError(name)
