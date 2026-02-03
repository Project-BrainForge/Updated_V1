"""
Count parameters for all neural-network models in this repo.

Models:
- CNN1Dpl (1D-CNN)
- HeckerLSTMpl (biLSTM)
- DeepSIFpl
- EEGViTpl (Transformer)
- STCNNTransformerpl (spatial CNN + Transformer)

Typical usage:
  python count_model_params.py --leadfield_mat D:/fyp/stESI_pub/anatomy/leadfield_75_20k.mat

Or specify dimensions explicitly:
  python count_model_params.py --n_electrodes 75 --n_sources 994 --n_times 500
"""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.io import loadmat

from models.cnn_1d import CNN1Dpl
from models.lstm import HeckerLSTMpl
from models.deepsif import DeepSIFpl
from models.vit import EEGViTpl
from models.st_cnn_transformer import STCNNTransformerpl


def _load_leadfield_mat(mat_path: str) -> np.ndarray:
    m = loadmat(mat_path)
    if "G" in m:
        return m["G"]
    if "fwd" in m:
        return m["fwd"]
    for k, v in m.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 2:
            return v
    raise KeyError(f"No leadfield matrix found in {mat_path}. Keys={list(m.keys())}")


def _count_params(module: torch.nn.Module) -> Tuple[int, int]:
    total = sum(int(p.numel()) for p in module.parameters())
    trainable = sum(int(p.numel()) for p in module.parameters() if p.requires_grad)
    return total, trainable


def _fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.3f}M"
    if n >= 1_000:
        return f"{n/1_000:.3f}K"
    return str(n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leadfield_mat", type=str, default=None, help="Optional .mat leadfield path to infer (E,S).")
    ap.add_argument("--n_electrodes", type=int, default=None, help="EEG electrodes (E).")
    ap.add_argument("--n_sources", type=int, default=None, help="Sources (S).")
    ap.add_argument("--n_times", type=int, default=500, help="Time samples (T) for transformer models.")

    # CNN1D hyperparams
    ap.add_argument("--cnn_inter_layer", type=int, default=2048, help="CNN1D intermediate channels.")
    ap.add_argument("--cnn_kernel_size", type=int, default=5, help="CNN1D kernel size.")

    # LSTM hyperparams
    ap.add_argument("--lstm_hidden_size", type=int, default=85, help="LSTM hidden size.")
    ap.add_argument("--lstm_mc_dropout", type=float, default=0.0, help="Enable MC-dropout LSTM if >0.")

    # DeepSIF hyperparams
    ap.add_argument("--deepsif_temporal_input_size", type=int, default=500)
    ap.add_argument("--deepsif_rnn_layer", type=int, default=3)

    # Transformer hyperparams (EEGViT + STCNNTransformer)
    ap.add_argument("--tr_embed_dim", type=int, default=256)
    ap.add_argument("--tr_depth", type=int, default=6)
    ap.add_argument("--tr_heads", type=int, default=8)
    ap.add_argument("--tr_mlp_dim", type=int, default=512)
    ap.add_argument("--tr_dropout", type=float, default=0.1)

    # STCNNTransformer spatial encoder hyperparams
    ap.add_argument("--st_spatial_hidden", type=int, default=64)
    ap.add_argument("--st_spatial_kernel", type=int, default=5)

    args = ap.parse_args()

    if args.leadfield_mat:
        lf = _load_leadfield_mat(args.leadfield_mat)
        lf = np.asarray(lf)
        if lf.ndim != 2:
            raise ValueError(f"Leadfield must be 2D, got shape={lf.shape} from {args.leadfield_mat}")
        lf_e, lf_s = int(lf.shape[0]), int(lf.shape[1])
        if args.n_electrodes is None:
            args.n_electrodes = lf_e
        if args.n_sources is None:
            args.n_sources = lf_s

    if args.n_electrodes is None or args.n_sources is None:
        raise SystemExit(
            "Provide either --leadfield_mat or both --n_electrodes and --n_sources."
        )

    E = int(args.n_electrodes)
    S = int(args.n_sources)
    T = int(args.n_times)

    models: Dict[str, torch.nn.Module] = {}

    # CNN1D (Lightning wrapper)
    models["cnn_1d"] = CNN1Dpl(
        channels=[E, int(args.cnn_inter_layer), S],
        kernel_size=int(args.cnn_kernel_size),
        bias=False,
        optimizer=None,
        lr=1e-3,
        criterion=None,
    )

    # LSTM (Lightning wrapper)
    models["lstm"] = HeckerLSTMpl(
        n_electrodes=E,
        hidden_size=int(args.lstm_hidden_size),
        n_sources=S,
        bias=False,
        optimizer=None,
        lr=1e-3,
        criterion=None,
        mc_dropout_rate=float(args.lstm_mc_dropout),
    )

    # DeepSIF (Lightning wrapper)
    models["deep_sif"] = DeepSIFpl(
        num_sensor=E,
        num_source=S,
        temporal_input_size=int(args.deepsif_temporal_input_size),
        rnn_layer=int(args.deepsif_rnn_layer),
        optimizer=None,
        lr=1e-3,
        criterion=None,
    )

    # EEGViT (Lightning wrapper)
    models["eeg_vit"] = EEGViTpl(
        num_sensor=E,
        num_source=S,
        n_times=T,
        embed_dim=int(args.tr_embed_dim),
        depth=int(args.tr_depth),
        num_heads=int(args.tr_heads),
        mlp_dim=int(args.tr_mlp_dim),
        dropout=float(args.tr_dropout),
        optimizer=None,
        lr=1e-3,
        criterion=None,
    )

    # STCNNTransformer (Lightning wrapper)
    models["stct"] = STCNNTransformerpl(
        num_sensor=E,
        num_source=S,
        n_times=T,
        embed_dim=int(args.tr_embed_dim),
        depth=int(args.tr_depth),
        num_heads=int(args.tr_heads),
        mlp_dim=int(args.tr_mlp_dim),
        dropout=float(args.tr_dropout),
        spatial_hidden_channels=int(args.st_spatial_hidden),
        spatial_kernel_size=int(args.st_spatial_kernel),
        optimizer=None,
        lr=1e-3,
        criterion=None,
    )

    print(f"Dimensions: E={E}, S={S}, T={T}")
    print("")
    print(f"{'model':<10} {'trainable':>14} {'total':>14}")
    print("-" * 42)
    for name, m in models.items():
        total, trainable = _count_params(m)
        print(f"{name:<10} {_fmt(trainable):>14} {_fmt(total):>14}")


if __name__ == "__main__":
    main()

