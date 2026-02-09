"""
Convert EDF EEG recordings to MAT files compatible with this repo's eval scripts.

Why this exists
---------------
`inverse_problem/eval_real.py`, `eval_real_cnn1d.py`, and `eval_real_lstm.py` can load "real" EEG
from a folder of `.mat` files (argument `-real_data_dir`). Those scripts expect each `.mat` file
to contain a 2D array named `eeg_data` with shape (n_electrodes, n_times).

`inverse_problem/eval_real_vit.py` currently loads the key `data` instead, so this converter saves
both keys (`eeg_data` and `data`) pointing to the same array for maximum compatibility.

Typical usage
-------------
1) Convert EDFs to 1-second windows (n_times=500 at fs=500) into `real_data/`:

    python tools/convert_edf_to_mat.py ^
        --input_dir eeg_real_mri_data ^
        --output_dir real_data ^
        --resample_fs 500 ^
        --window 500 ^
        --stride 500

2) If you want to force the output to 75 channels (leadfield_75_20k.mat) by mapping your EDF
   channel names onto the electrode list stored in `anatomy/electrode_75.mat` (missing channels
   are filled with zeros):

    python tools/convert_edf_to_mat.py ^
        --input_dir eeg_real_mri_data ^
        --output_dir real_data ^
        --resample_fs 500 ^
        --window 500 ^
        --stride 500 ^
        --target_electrodes_mat anatomy/electrode_75.mat

Notes
-----
- If your EDF contains fewer electrodes than the leadfield used by a model (e.g., 19 vs 75),
  evaluation scripts will error unless you either (a) use a leadfield with matching electrodes,
  or (b) use `--target_electrodes_mat` to expand/map channels (zeros for missing channels).
- This script requires `mne` and `scipy` (both are listed in this repo's requirements).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _norm_chan(name: str) -> str:
    """
    Normalize channel names for matching across toolboxes.

    Examples:
      "EEG Fp1-REF" -> "FP1"
      "Fp2"         -> "FP2"
      "T3"          -> "T3"
      "A1"          -> "A1"
    """
    if not isinstance(name, str):
        name = str(name)
    n = name.strip()
    # Common prefixes in EDF exports
    for p in ("EEG ", "EEG_", "EEG-", "EEG:"):
        if n.upper().startswith(p):
            n = n[len(p) :]
            break
    # Drop reference suffixes
    for suf in ("-REF", "-LE", "-AVG", "-M1", "-M2", "-A1", "-A2"):
        if n.upper().endswith(suf):
            n = n[: -len(suf)]
            break
    # Keep only alphanumerics
    n = "".join([c for c in n if c.isalnum()])
    return n.upper()


@dataclass
class TargetElectrodes:
    names: List[str]
    norm_to_index: Dict[str, int]


def _load_target_electrodes_from_mat(mat_path: Path) -> TargetElectrodes:
    """
    Load target electrode names from `anatomy/electrode_75.mat`.

    The file is expected to contain a variable like `eloc75` (Matlab struct array)
    with a `labels` or `label` / `labels` field (depending on exporter).

    This function is intentionally defensive: it tries a few common patterns.
    """
    try:
        from scipy.io import loadmat
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "scipy is required to load .mat electrode lists. Install with: pip install scipy"
        ) from e

    m = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    keys = [k for k in m.keys() if not k.startswith("__")]
    if not keys:
        raise ValueError(f"No variables found in {mat_path}")

    # Prefer known variable names
    candidates = []
    for k in keys:
        if k.lower() in ("eloc75", "eloc", "electrodes", "chanlocs", "chanloc"):
            candidates.append(k)
    candidates += keys

    labels: Optional[List[str]] = None
    last_err: Optional[Exception] = None

    for k in candidates:
        v = m.get(k, None)
        if v is None:
            continue

        try:
            # Case 1: struct array with attribute `.labels` or `.label`
            if hasattr(v, "labels"):
                raw = getattr(v, "labels")
                labels = _to_string_list(raw)
                break
            if hasattr(v, "label"):
                raw = getattr(v, "label")
                labels = _to_string_list(raw)
                break

            # Case 2: object array of structs, each has `.labels`/`.label`/`.labels`
            if isinstance(v, np.ndarray) and v.dtype == object and v.size > 0:
                first = v.flat[0]
                for field in ("labels", "label", "name", "names"):
                    if hasattr(first, field):
                        labels = [_to_string(getattr(x, field)) for x in v.flat]
                        break
                if labels:
                    break
        except Exception as e:
            last_err = e
            continue

    if not labels:
        msg = (
            f"Could not extract electrode labels from {mat_path}. "
            f"Available keys: {keys}"
        )
        if last_err:
            msg += f"\nLast error: {last_err}"
        raise ValueError(msg)

    names = [str(x) for x in labels]
    norm_to_index = {_norm_chan(n): i for i, n in enumerate(names)}
    return TargetElectrodes(names=names, norm_to_index=norm_to_index)


def _to_string(x) -> str:
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8", errors="ignore")
        except Exception:
            return str(x)
    if isinstance(x, np.str_):
        return str(x)
    return str(x)


def _to_string_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [_to_string(xx) for xx in x]
    if isinstance(x, np.ndarray):
        return [_to_string(xx) for xx in x.flat]
    return [_to_string(x)]


def _iter_edf_files(input_dir: Path) -> List[Path]:
    if input_dir.is_file():
        return [input_dir]
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")
    files = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".edf":
            files.append(p)
    return sorted(files)


def _window_indices(n_samples: int, window: int, stride: int) -> Iterable[Tuple[int, int]]:
    if window <= 0:
        yield (0, n_samples)
        return
    if stride <= 0:
        stride = window
    if n_samples <= window:
        yield (0, window)
        return
    for start in range(0, n_samples - window + 1, stride):
        yield (start, start + window)


def _pad_or_crop(x: np.ndarray, n_time: int) -> np.ndarray:
    """Ensure time dimension equals n_time by padding with zeros or cropping."""
    if x.shape[1] == n_time:
        return x
    if x.shape[1] > n_time:
        return x[:, :n_time]
    pad = np.zeros((x.shape[0], n_time - x.shape[1]), dtype=x.dtype)
    return np.concatenate([x, pad], axis=1)


def _map_to_target(
    data: np.ndarray, ch_names: List[str], target: TargetElectrodes
) -> Tuple[np.ndarray, List[str], Dict[str, str]]:
    """
    Map input channels onto target electrode ordering.
    Missing channels stay as zeros. Returns mapping info for debugging.
    """
    out = np.zeros((len(target.names), data.shape[1]), dtype=data.dtype)
    mapping: Dict[str, str] = {}
    used = set()

    for src_idx, src_name in enumerate(ch_names):
        nn = _norm_chan(src_name)
        if nn in target.norm_to_index:
            dst_idx = target.norm_to_index[nn]
            out[dst_idx, :] = data[src_idx, :]
            mapping[target.names[dst_idx]] = src_name
            used.add(dst_idx)

    return out, target.names, mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="eeg_real_mri_data",
        help="Folder containing EDF files (or a single EDF file path).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="real_data",
        help="Folder where MAT windows will be written.",
    )
    parser.add_argument(
        "--resample_fs",
        type=float,
        default=500.0,
        help="Resample EDF to this sampling rate (Hz). Set <=0 to keep original.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=500,
        help="Number of time samples per exported window (e.g., 500). Use <=0 for full signal only.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=500,
        help="Stride (samples) between consecutive windows.",
    )
    parser.add_argument(
        "--start_sec",
        type=float,
        default=0.0,
        help="Start time (seconds) to begin exporting from the EDF.",
    )
    parser.add_argument(
        "--stop_sec",
        type=float,
        default=-1.0,
        help="Stop time (seconds). Use <0 to export until end.",
    )
    parser.add_argument(
        "--car",
        action="store_true",
        help="Apply common average reference (subtract mean across channels at each time).",
    )
    parser.add_argument(
        "--demean",
        action="store_true",
        help="Remove per-channel mean (over time).",
    )
    parser.add_argument(
        "--save_full",
        action="store_true",
        help="Also save one *_full.mat per EDF containing the full (cropped) continuous signal.",
    )
    parser.add_argument(
        "--target_electrodes_mat",
        type=str,
        default=None,
        help=(
            "Optional path to a .mat containing the target electrode list (e.g. anatomy/electrode_75.mat). "
            "If set, output will be mapped to this electrode ordering; missing channels are filled with zeros."
        ),
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="eeg_and_src_data",
        help="Prefix for output MAT files (default matches eval_real.py glob).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target: Optional[TargetElectrodes] = None
    if args.target_electrodes_mat:
        target = _load_target_electrodes_from_mat(Path(args.target_electrodes_mat))

    try:
        import mne
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "mne is required to read EDF files. Install with: pip install mne"
        ) from e

    try:
        from scipy.io import savemat
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "scipy is required to write .mat files. Install with: pip install scipy"
        ) from e

    edf_files = _iter_edf_files(input_dir)
    if not edf_files:
        raise FileNotFoundError(f"No .edf files found under: {input_dir}")

    global_idx = 0
    for edf_path in edf_files:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

        # Prefer EEG channels if they are typed as such; otherwise keep everything.
        eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=False, eog=False, ecg=False)
        if len(eeg_picks) > 0:
            raw.pick(eeg_picks)

        if args.resample_fs and args.resample_fs > 0:
            raw.resample(float(args.resample_fs), npad="auto")

        fs = float(raw.info["sfreq"])
        start_samp = int(round(max(0.0, float(args.start_sec)) * fs))
        if args.stop_sec and args.stop_sec > 0:
            stop_samp = int(round(float(args.stop_sec) * fs))
        else:
            stop_samp = raw.n_times

        start_samp = max(0, min(start_samp, raw.n_times))
        stop_samp = max(0, min(stop_samp, raw.n_times))
        if stop_samp <= start_samp:
            raise ValueError(f"Invalid crop for {edf_path}: start={start_samp}, stop={stop_samp}")

        data = raw.get_data(start=start_samp, stop=stop_samp).astype(np.float32)  # (C, T)
        ch_names = list(raw.ch_names)

        if args.demean:
            data = data - np.mean(data, axis=1, keepdims=True)
        if args.car:
            data = data - np.mean(data, axis=0, keepdims=True)

        mapping: Optional[Dict[str, str]] = None
        if target is not None:
            data, ch_names, mapping = _map_to_target(data, ch_names, target)

        # Optionally save continuous version
        if args.save_full:
            full_path = out_dir / f"{edf_path.stem}_full.mat"
            savemat(
                str(full_path),
                {
                    "eeg_data": data,
                    "data": data,  # compatibility with eval_real_vit.py
                    "fs": fs,
                    "sfreq": fs,
                    "ch_names": np.array(ch_names, dtype=object),
                    "source_file": str(edf_path),
                    "start_sec": float(args.start_sec),
                    "stop_sec": float(args.stop_sec),
                    "start_samp": int(start_samp),
                    "stop_samp": int(stop_samp),
                    "channel_mapping": mapping or {},
                },
                do_compression=True,
            )

        # Export windows compatible with eval_real*.py
        w = int(args.window)
        s = int(args.stride)
        for (a, b) in _window_indices(data.shape[1], w, s):
            seg = data[:, a:b]
            if w > 0:
                seg = _pad_or_crop(seg, w)

            out_path = out_dir / f"{args.prefix}_{global_idx}.mat"
            savemat(
                str(out_path),
                {
                    "eeg_data": seg,
                    "data": seg,  # compatibility with eval_real_vit.py
                    "fs": fs,
                    "sfreq": fs,
                    "ch_names": np.array(ch_names, dtype=object),
                    "source_file": str(edf_path),
                    "window_start_samp": int(a + start_samp),
                    "window_stop_samp": int(min(b + start_samp, stop_samp)),
                    "window_start_sec": float((a + start_samp) / fs),
                    "window_stop_sec": float((min(b + start_samp, stop_samp)) / fs),
                    "channel_mapping": mapping or {},
                },
                do_compression=True,
            )
            global_idx += 1

    print(f"Done. Wrote {global_idx} MAT windows to: {out_dir}")


if __name__ == "__main__":
    main()

