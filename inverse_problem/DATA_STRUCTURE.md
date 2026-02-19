# Simulation and loader data structures

## In `simu_extended_source.py`

- **Patch**: One “extended source” = a set of **neighboring region indices** that are active together (one spatial blob). E.g. `patch = [45, 12, 88, 3]` = regions 45, 12, 88, 3 form one patch.

- **patches**: Dict `{'patch_1': [r1, r2, ...], 'patch_2': [r3, r4, ...], ...}`. One key per extended source; each value is the list of **region indices** (source-space indices) in that patch.

- **n_patch**: Number of extended sources (patches) in that sample. E.g. `n_patch = 2` → two blobs.

- **act_src**: Same as `patches`; saved in metadata JSON as `act_src` (e.g. for evaluation).

---

## In the loader saved `.mat` file

- **eeg_data**: `(n_electrodes, n_times)` — EEG for that sample.

- **src_data**: `(n_sources, n_times)` — full source-space activity. **Only rows at active region indices are non-zero**; all other rows are zero.

- **region_data** (or **regions**): `(max_regions_per_patch, n_patch)`. Column `j` = region indices of patch `j`, **zero-padded** to the same length. So it tells you “which regions belong to which patch” (spatial grouping).

- **active_regions**: `(n_active,)` — **1D array of region indices that have non-zero rows in `src_data`** (no zero padding). This is the list of **source regions that actually generated the EEG**: the union of all patches, sorted and unique.

---

## Getting “source regions of the EEG”

- **From the .mat file**: use **`active_regions`**. Those are exactly the region indices where `src_data` is non-zero.
- **Equivalently**: take `src_data`, find row indices where the row is not all zeros → same set as `active_regions`.
