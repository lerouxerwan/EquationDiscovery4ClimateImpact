from pathlib import Path

import xarray as xr

from utils.utils_path import DATA_PATH

MASK_PATH = Path(DATA_PATH) / "mask"


def compute_weights(mask_name: str='LION4') -> xr.DataArray:
    # Compute all weights
    mesh_mask_MED = xr.open_dataset(MASK_PATH / "mesh_mask_MED12_v3.6_75lev.nc")
    lat = mesh_mask_MED['e2t'][0, :, :]
    lon = mesh_mask_MED['e1t'][0, :, :]
    weights = lat * lon
    # Keep only weights for the mask of interest
    mask = compute_mask(mask_name)
    weights = weights.where(mask, drop=False)
    weights = weights.fillna(0)
    return weights

def compute_mask(mask_name: str='LION4') -> xr.DataArray:
    return xr.open_dataset(MASK_PATH / "subbasins_dev_MED12.nc")[mask_name].fillna(False)


