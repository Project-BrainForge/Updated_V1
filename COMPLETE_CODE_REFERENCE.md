# Complete code reference (simulation + loader + data structure)

Below are the full contents of the modified files. Paths are relative to the repo root.

---

## 1. `simu_source_python/simu_extended_source.py`

See the file on disk; it is **449 lines**. Summary of behavior:

- **Args:** `-sin`, `-ne`, `-mk`, `-ss`, `-o`, `-sn`, `-rf`, `--leadfield_mat`, `-fs`, `-d`, `-m`, `-np_min`, `-np_max`, `-o_min`, `-o_max`, **`-max_active` / `--max_active_regions` (default 70)**, temporal args, `-ds`.
- **994 regions:** `_compute_knn_neighbors(spos, k=70)`.
- **Mesh:** neighbors limited to 70 per vertex; then patch loop caps **total active regions per sample** at `max_active_regions` (truncate patch by distance to seed if needed); `n_patch` set to actual number of patches after loop.

Full path: **`/media/pasindu/DATA/fyp/stESI_pub/simu_source_python/simu_extended_source.py`**

---

## 2. `simu_source_python/utils.py`

See the file on disk; it is **148 lines**. Summary:

- **`get_component_extended_src(..., patch_override=None)`**  
  If `patch_override` is given (1D array of region indices), that is used as the patch instead of `get_patch(...)`. Defines `patch_dim = 0.0` when the patch has only one source.

Full path: **`/media/pasindu/DATA/fyp/stESI_pub/simu_source_python/utils.py`**

---

## 3. `inverse_problem/loaders.py` — `EsiDatasetds_new.__getitem__` (save .mat block)

This block builds **region_data**, **active_regions**, and saves the .mat file (around lines 165–192):

```python
        # save as mat file with index as the file name
        # Build region matrix: one column per patch, rows = source indices (0-padded)
        act_src = self.md_dict[self.ids[index]].get("act_src", {})
        patches_list = [
            act_src[k]
            for k in sorted(act_src.keys(), key=lambda x: int(x.split("_")[-1]))
        ]
        if patches_list:
            max_len = max(len(p) for p in patches_list)
            region_data = np.zeros((max_len, len(patches_list)), dtype=np.int64)
            for col, p in enumerate(patches_list):
                region_data[: len(p), col] = p
            # active_regions = source region indices with non-zero rows in src_data (no zero padding)
            active_regions = np.unique(np.concatenate(patches_list)).astype(np.int64)
        else:
            region_data = np.zeros((0, 0), dtype=np.int64)
            active_regions = np.array([], dtype=np.int64)
        eeg_np = eeg_data.numpy() if hasattr(eeg_data, "numpy") else np.asarray(eeg_data)
        src_np = src_data.numpy() if hasattr(src_data, "numpy") else np.asarray(src_data)
        savemat(
            f"eeg_and_src_data_{index}.mat",
            {
                "eeg_data": eeg_np,
                "src_data": src_np,
                "region_data": region_data,
                "active_regions": active_regions,
            },
        )

        return eeg_data, src_data
```

Full path: **`/media/pasindu/DATA/fyp/stESI_pub/inverse_problem/loaders.py`**

---

## 4. `inverse_problem/DATA_STRUCTURE.md`

Full path: **`/media/pasindu/DATA/fyp/stESI_pub/inverse_problem/DATA_STRUCTURE.md`**

```markdown
# Simulation and loader data structures

## In `simu_extended_source.py`

- **Patch**: One "extended source" = a set of **neighboring region indices** that are active together (one spatial blob). E.g. `patch = [45, 12, 88, 3]` = regions 45, 12, 88, 3 form one patch.

- **patches**: Dict `{'patch_1': [r1, r2, ...], 'patch_2': [r3, r4, ...], ...}`. One key per extended source; each value is the list of **region indices** (source-space indices) in that patch.

- **n_patch**: Number of extended sources (patches) in that sample. E.g. `n_patch = 2` → two blobs.

- **act_src**: Same as `patches`; saved in metadata JSON as `act_src` (e.g. for evaluation).

---

## In the loader saved `.mat` file

- **eeg_data**: `(n_electrodes, n_times)` — EEG for that sample.

- **src_data**: `(n_sources, n_times)` — full source-space activity. **Only rows at active region indices are non-zero**; all other rows are zero.

- **region_data** (or **regions**): `(max_regions_per_patch, n_patch)`. Column `j` = region indices of patch `j`, **zero-padded** to the same length. So it tells you "which regions belong to which patch" (spatial grouping).

- **active_regions**: `(n_active,)` — **1D array of region indices that have non-zero rows in `src_data`** (no zero padding). This is the list of **source regions that actually generated the EEG**: the union of all patches, sorted and unique.

---

## Getting "source regions of the EEG"

- **From the .mat file**: use **`active_regions`**. Those are exactly the region indices where `src_data` is non-zero.
- **Equivalently**: take `src_data`, find row indices where the row is not all zeros → same set as `active_regions`.
```

---

## Quick run commands

**Generate data (max 70 active regions per sample, 70 neighbors for 994):**
```bash
python simu_source_python/simu_extended_source.py -sin mes_debug_python -ne 100 -mk standard_1020 -ss fsav_994 -o constrained -sn fsaverage -rf /media/pasindu/DATA/fyp/stESI_pub --leadfield_mat /media/pasindu/DATA/fyp/stESI_pub/anatomy/leadfield_75_20k.mat -fs 500 -d 1000
```
Optional: `-max_active 50` to cap at 50 active regions per sample.

**Train (from repo root):**
```bash
cd inverse_problem && python main_train.py mes_debug_python -simu_type sereega -root_simu /media/pasindu/DATA/fyp/stESI_pub -results_path /media/pasindu/DATA/fyp/stESI_pub/inverse_problem/results -source_space fsav_994 -model DEEPSIF -to_load 100 -eeg_snr 5 -n_epochs 100 -scaler linear -loss cosine -sfolder train_size_impact -leadfield_mat /media/pasindu/DATA/fyp/stESI_pub/anatomy/leadfield_75_20k.mat -deepsif_temporal_input_size 500
```
