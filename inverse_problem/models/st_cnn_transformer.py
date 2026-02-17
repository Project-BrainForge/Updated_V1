"""
Spatio-temporal model: CNN (spatial) + Transformer (temporal)

Goal: EEG (B, E, T) -> Sources (B, S, T)

- Spatial encoder (CNN): for each time step, apply 1D convolutions over the electrode axis
  to extract an embedding vector (token).
- Temporal encoder (Transformer): run a TransformerEncoder over the sequence of tokens (time).
- Output head: project each time token to sources.
"""

from __future__ import annotations

import torch
from torch import nn
import pytorch_lightning as pl


class SpatialCNN(nn.Module):
    """
    Encode per-time-step electrode topography (E,) into an embedding (D,) using 1D convs over electrodes.

    Input to forward: x shaped (B, E, T)
    Output: tokens shaped (B, T, D)
    """

    def __init__(
        self,
        num_sensor: int,
        embed_dim: int = 256,
        hidden_channels: int = 64,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_sensor = int(num_sensor)
        self.embed_dim = int(embed_dim)

        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=kernel_size, padding=pad, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, embed_dim, kernel_size=kernel_size, padding=pad, bias=False),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, E, T) -> (B, T, E)
        x = x.permute(0, 2, 1)
        b, t, e = x.shape
        if e != self.num_sensor:
            raise ValueError(f"SpatialCNN expected E={self.num_sensor}, got E={e}")

        # treat each time step independently: (B*T, 1, E)
        x = x.reshape(b * t, 1, e)
        x = self.net(x)  # (B*T, D, E)
        x = self.pool(x).squeeze(-1)  # (B*T, D)
        return x.reshape(b, t, self.embed_dim)  # (B, T, D)


class STCNNTransformer(nn.Module):
    def __init__(
        self,
        num_sensor: int,
        num_source: int,
        n_times: int = 500,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        spatial_hidden_channels: int = 64,
        spatial_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.num_sensor = int(num_sensor)
        self.num_source = int(num_source)
        self.n_times = int(n_times)

        self.spatial = SpatialCNN(
            num_sensor=num_sensor,
            embed_dim=embed_dim,
            hidden_channels=spatial_hidden_channels,
            kernel_size=spatial_kernel_size,
            dropout=dropout,
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_times, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.out_proj = nn.Linear(embed_dim, self.num_source)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, E, T)
        returns: (B, S, T)
        """
        if x.shape[-1] != self.n_times:
            raise ValueError(
                f"STCNNTransformer was initialized with n_times={self.n_times}, got T={x.shape[-1]}."
            )
        tokens = self.spatial(x)  # (B, T, D)
        tokens = self.pos_drop(tokens + self.pos_embed)
        tokens = self.encoder(tokens)  # (B, T, D)
        out = self.out_proj(tokens)  # (B, T, S)
        return out.permute(0, 2, 1)  # (B, S, T)


class STCNNTransformerpl(pl.LightningModule):
    def __init__(
        self,
        num_sensor: int,
        num_source: int,
        n_times: int = 500,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        spatial_hidden_channels: int = 64,
        spatial_kernel_size: int = 5,
        optimizer=torch.optim.Adam,
        lr: float = 1e-3,
        criterion=torch.nn.MSELoss(),
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["criterion", "optimizer"])

        self.optimizer = optimizer
        self.lr = lr
        self.criterion = criterion

        self.model = STCNNTransformer(
            num_sensor=num_sensor,
            num_source=num_source,
            n_times=n_times,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            spatial_hidden_channels=spatial_hidden_channels,
            spatial_kernel_size=spatial_kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        eeg, src = batch
        eeg = eeg.float()
        src = src.float()
        src_hat = self.forward(eeg)
        loss = self.criterion(src_hat, src)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        eeg, src = batch
        eeg = eeg.float()
        src = src.float()
        src_hat = self.forward(eeg)
        loss = self.criterion(src_hat, src)
        self.log("validation_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

