"""
Evaluate ST-CNN-Transformer on EEG-only data and export predictions (`all_out`).

Model: spatial CNN (over electrodes) + temporal Transformer (over time).

Saves `all_out` to a .mat file with shape:
  (n_samples, n_times, n_sources)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
from typing import List

import numpy as np
import torch
from scipy.io import loadmat, savemat
from pytorch_lightning import seed_everything

from load_data.utl_data import load_eeg_data_from_file, get_matching_info
from utils import utl
from models.st_cnn_transformer import STCNNTransformerpl


def _pick_model_path_from_run_dir(run_dir: str) -> str:
    trained_models_dir = os.path.join(run_dir, "trained_models")
    candidates = [
        os.path.join(trained_models_dir, "STCT_model.pt"),
        os.path.join(trained_models_dir, "stct_model.pt"),
        os.path.join(trained_models_dir, "STCNNTRANSFORMER_model.pt"),
        os.path.join(trained_models_dir, "stcnntransformer_model.pt"),
        os.path.join(trained_models_dir, "CNNTRANSFORMER_model.pt"),
        os.path.join(trained_models_dir, "cnntransformer_model.pt"),
        os.path.join(trained_models_dir, "ST_CNN_TRANSFORMER_model.pt"),
        os.path.join(trained_models_dir, "st_cnn_transformer_model.pt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not locate STCT weights under: {trained_models_dir}. "
        "Pass -weights_path to specify the .pt file explicitly."
    )


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


def _resolve_root_base(root_simu: Path, subject_name: str) -> Path:
    if (root_simu / "simulation" / subject_name).is_dir():
        return root_simu / "simulation" / subject_name
    return root_simu


def _build_general_config(
    simu_name: str,
    orientation: str,
    electrode_montage: str,
    source_space: str,
    n_times: int,
    n_sources: int,
    n_electrodes: int,
) -> dict:
    return {
        "simu_name": simu_name,
        "eeg_snr": "infdb",
        "source_space": {
            "constrained_orientation": orientation == "constrained",
            "src_sampling": source_space,
            "n_sources": int(n_sources),
        },
        "electrode_space": {
            "electrode_montage": electrode_montage,
            "n_electrodes": int(n_electrodes),
        },
        "rec_info": {"n_times": int(n_times), "fs": 1},
    }


def _load_eeg_matrix(path: str, n_times: int) -> np.ndarray:
    eeg = load_eeg_data_from_file(path)[0][0]
    eeg = np.asarray(eeg, dtype=np.float32)
    if eeg.ndim == 2 and eeg.shape[0] > eeg.shape[1]:
        eeg = eeg.T
    if eeg.shape[1] < n_times:
        pad = np.zeros((eeg.shape[0], n_times - eeg.shape[1]), dtype=eeg.dtype)
        eeg = np.concatenate([eeg, pad], axis=1)
    elif eeg.shape[1] > n_times:
        eeg = eeg[:, :n_times]
    return eeg


def _natural_key(p: Path) -> tuple:
    m = re.search(r"(\d+)(?!.*\d)", p.stem)
    if m:
        return (p.stem[: m.start()], int(m.group(1)), p.suffix)
    return (p.stem, -1, p.suffix)


def _iter_real_mat_paths(real_data_dir: str, pattern: str) -> List[Path]:
    d = Path(real_data_dir)
    if not d.exists():
        raise FileNotFoundError(f"real_data_dir does not exist: {real_data_dir}")
    if d.is_file():
        return [d]
    mats = sorted(d.glob(pattern), key=_natural_key)
    if not mats:
        raise FileNotFoundError(
            f"No .mat files found in {real_data_dir} matching pattern {pattern!r}"
        )
    return mats


def _load_real_eeg_from_mat(mat_path: str, n_times: int) -> np.ndarray:
    print("Loading EEG data from mat file: ", mat_path)
    m = loadmat(mat_path)
    if "eeg_data" not in m:
        raise KeyError(f"Missing key 'eeg_data' in {mat_path}. Keys={list(m.keys())}")
    eeg = np.asarray(m["eeg_data"], dtype=np.float32).squeeze()
    if eeg.ndim != 2:
        raise ValueError(f"Expected eeg_data to be 2D in {mat_path}, got shape={eeg.shape}")
    if eeg.shape[1] != n_times and eeg.shape[0] == n_times:
        eeg = eeg.T
    if eeg.shape[1] < n_times:
        pad = np.zeros((eeg.shape[0], n_times - eeg.shape[1]), dtype=eeg.dtype)
        eeg = np.concatenate([eeg, pad], axis=1)
    elif eeg.shape[1] > n_times:
        eeg = eeg[:, :n_times]
    return eeg


def main() -> None:
    seed_everything(0)

    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("simu_name", type=str, help="Simulation name (folder name)")
    parser.add_argument("-root_simu", type=str, required=False)
    parser.add_argument("-subject_name", type=str, default="fsaverage")
    parser.add_argument("-orientation", type=str, default="constrained")
    parser.add_argument("-electrode_montage", type=str, default="standard_1020")
    parser.add_argument("-source_space", type=str, default="fsav_994")
    parser.add_argument("-n_times", type=int, default=500)
    parser.add_argument("-to_load", type=int, default=-1)
    parser.add_argument("-real_data_dir", type=str, default=None)
    parser.add_argument("-real_data_glob", type=str, default="eeg_and_src_data_*.mat")

    parser.add_argument("-leadfield_mat", type=str, required=True)
    parser.add_argument("-train_run_dir", type=str, required=True)
    parser.add_argument("-weights_path", type=str, default=None)
    parser.add_argument(
        "-ckpt_path",
        type=str,
        default=None,
        help=(
            "Optional PyTorch-Lightning checkpoint (.ckpt). "
            "If provided, the model is loaded from this checkpoint and -weights_path is ignored."
        ),
    )

    parser.add_argument("-st_embed_dim", type=int, default=256)
    parser.add_argument("-st_depth", type=int, default=6)
    parser.add_argument("-st_heads", type=int, default=8)
    parser.add_argument("-st_mlp_dim", type=int, default=512)
    parser.add_argument("-st_dropout", type=float, default=0.1)
    parser.add_argument("-st_spatial_hidden", type=int, default=64)
    parser.add_argument("-st_spatial_kernel", type=int, default=5)

    parser.add_argument("-batch_size", type=int, default=8)
    parser.add_argument("--no_gfp_scaling", action="store_true")
    parser.add_argument("-out_mat", type=str, default=None)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    fwd = np.asarray(_load_leadfield_mat(args.leadfield_mat), dtype=np.float32)
    n_electrodes, n_sources = int(fwd.shape[0]), int(fwd.shape[1])

    use_real_mats = args.real_data_dir is not None
    if use_real_mats:
        mat_paths = _iter_real_mat_paths(args.real_data_dir, args.real_data_glob)
        if args.to_load and args.to_load > 0:
            mat_paths = mat_paths[: args.to_load]
        n_samples = len(mat_paths)
        print(f"Loading {n_samples} EEG samples from real_data mats")
    else:
        if not args.root_simu:
            parser.error(
                "Either provide -root_simu (simulation mode) or -real_data_dir (real_data mode)."
            )

        root_base = _resolve_root_base(Path(args.root_simu), args.subject_name)
        data_folder_name = os.path.normpath(
            os.path.join(
                str(root_base),
                args.orientation,
                args.electrode_montage,
                args.source_space,
                "simu",
            )
        )
        general_cfg = _build_general_config(
            simu_name=args.simu_name,
            orientation=args.orientation,
            electrode_montage=args.electrode_montage,
            source_space=args.source_space,
            n_times=args.n_times,
            n_sources=n_sources,
            n_electrodes=n_electrodes,
        )
        ids, eeg_dict, _, _ = get_matching_info(
            data_folder_name=data_folder_name,
            general_config_dict=general_cfg,
            root_simu=str(root_base),
        )
        if args.to_load and args.to_load > 0:
            ids = ids[: args.to_load]
        n_samples = len(ids)
        print(f"Loading {n_samples} EEG samples from simulation match JSON")

    if args.ckpt_path:
        print(f"Loading checkpoint: {args.ckpt_path}")
        model = STCNNTransformerpl.load_from_checkpoint(
            args.ckpt_path,
            map_location=torch.device("cpu"),
        )
    else:
        weights_path = args.weights_path or _pick_model_path_from_run_dir(args.train_run_dir)
        print(f"Using weights: {weights_path}")

        model = STCNNTransformerpl(
            num_sensor=n_electrodes,
            num_source=n_sources,
            n_times=args.n_times,
            embed_dim=args.st_embed_dim,
            depth=args.st_depth,
            num_heads=args.st_heads,
            mlp_dim=args.st_mlp_dim,
            dropout=args.st_dropout,
            spatial_hidden_channels=args.st_spatial_hidden,
            spatial_kernel_size=args.st_spatial_kernel,
            optimizer=None,
            lr=1e-3,
            criterion=None,
        )
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)

    model.eval()
    model.to(device)

    all_out = np.zeros((n_samples, args.n_times, n_sources), dtype=np.float32)

    with torch.no_grad():
        for i in range(n_samples):
            if use_real_mats:
                eeg = _load_real_eeg_from_mat(str(mat_paths[i]), args.n_times)
            else:
                eeg = _load_eeg_matrix(eeg_dict[ids[i]], args.n_times)

            # model expects (B,E,T)
            eeg_t = torch.from_numpy(eeg).unsqueeze(0).to(device)
            out = model(eeg_t).squeeze(0)  # (S,T)

            if not args.no_gfp_scaling:
                out = utl.gfp_scaling(
                    torch.from_numpy(eeg).to(out.device),
                    out,
                    torch.from_numpy(fwd).to(out.device),
                )

            all_out[i] = out.permute(1, 0).detach().cpu().numpy()  # (T,S)

            if (i + 1) % 50 == 0:
                print(f"Processed {i+1}/{n_samples}")

    out_path = args.out_mat or os.path.join(args.train_run_dir, "eval_real_all_out_stct.mat")
    savemat(out_path, {"all_out": all_out})
    print(f"Saved all_out to: {out_path}")


if __name__ == "__main__":
    main()

