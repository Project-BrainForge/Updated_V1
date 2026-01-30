# Data Conversion Scripts

This directory contains scripts to prepare data for the brain visualization web app.

## convert_mat_to_json.py

Converts MATLAB brain mesh data to JSON format for web visualization.

### Usage

Basic conversion:
```bash
python convert_mat_to_json.py
```

With custom paths:
```bash
python convert_mat_to_json.py \
  --brain-mesh ../anatomy/fs_cortex_20k.mat \
  --region-mapping ../anatomy/fs_cortex_20k_region_mapping.mat \
  --output-dir ../public/data
```

With mesh decimation (to reduce file size):
```bash
python convert_mat_to_json.py --decimation 2
```

Export a sample prediction:
```bash
python convert_mat_to_json.py \
  --prediction /path/to/eval_real_all_out_vit_25.mat \
  --sample-index 0
```

### Output

The script generates:
- `brain_mesh.json` - Full brain mesh data
- `brain_mesh.json.gz` - Compressed version
- `prediction_sample_N.json` - Sample prediction data (if --prediction provided)

These files can be:
1. Placed in `public/data/` to be served statically
2. Loaded by the frontend for visualization without needing the backend API
