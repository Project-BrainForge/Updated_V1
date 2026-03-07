from loaders import EsiDatasetds_new
from load_data.FolderStructure import FolderStructure
import json, os

root = "/media/pasindu/DATA/fyp/stESI_pub/simulation/fsaverage"
cfg = f"{root}/constrained/standard_1020/fsav_994/simu/mes_debug_python/mes_debug_pythonfsav_994_config.json"

ds = EsiDatasetds_new(
    root_simu=root,
    config_file=cfg,
    simu_name="mes_debug_python",
    source_space="fsav_994",
    electrode_montage="standard_1020",
    to_load=2000,
    snr_db=5,
    noise_type={"white": 1.0, "pink": 0.0},
    norm="linear",
)

for i in range(len(ds)):
    _ = ds[i]  # triggers save