from pathlib import Path

import xarray as xr

from utils.utils_path import DATA_PATH


def compute_weights() -> xr.DataArray:
    MASK_PATH = Path(DATA_PATH) / "mask"
    # Compute all weights
    mesh_mask_MED = xr.open_dataset(MASK_PATH / "mesh_mask_MED12_v3.6_75lev.nc")
    lat = mesh_mask_MED['e2t'][0, :, :]
    lon = mesh_mask_MED['e1t'][0, :, :]
    weights = lat * lon
    # Keep only LION4 weights
    gol4_mask = xr.open_dataset(MASK_PATH / "subbasins_dev_MED12.nc")['LION4']
    gol4_mask = gol4_mask.fillna(False)
    weights = weights.where(gol4_mask, drop=False)
    weights = weights.fillna(0)
    return weights
