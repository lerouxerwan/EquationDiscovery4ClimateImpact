import os
from configparser import Interpolation
from pathlib import Path

import numpy as np
import xarray as xr

from projects.ensemble_30.shortwave.raw import get_da, get_filename
from projects.ensemble_30.utils_ensemble_30 import ENSEMBLE_30_PATH, WEBPATH
from utils.utils_bash_call import bash_call
from utils.utils_log import log_info


"""Download data for bilinear interpolation"""

INTERPOLATION_FILENAME = "weights_bilinear_MF_ALD3_MED12.nc"
INTERPOLATION_FILEPATH = ENSEMBLE_30_PATH / INTERPOLATION_FILENAME

def download() -> None:
    filepath = os.path.join(WEBPATH, INTERPOLATION_FILENAME)
    bash_call(f'wget -P {ENSEMBLE_30_PATH}/ {filepath}')


"""Apply bilinear interpolation"""

def extract_interpolated_da(year: int) -> xr.DataArray:
    """For a given year, interpolate shortwave to a MED12 grid
    Bilinear interpolation consists of 4 contributions (4 points nearby) that are summed"""
    log_info(f'Compute data array for year={year} for a MED12 grid')
    # Extract data needed for the interpolation
    interpolation_ds = xr.open_dataset(INTERPOLATION_FILEPATH)
    da = get_da(year).resample(time='1MS').mean(dim='time') # Resample to a month frequency
    values = da.values
    max_lon = 440 # values are of dimension 160x440
    # Build a list of contribution data array for the interpolation, one contribution for each point nearby
    contribution_das = []
    for index_point in range(1, 5):
        log_info(f'index point = {index_point}')

        source_da = interpolation_ds[f'src0{index_point}']
        # Matrix
        i_matrix = source_da.values // max_lon
        j_matrix = source_da.values % max_lon

        # Fill contribution data array with raw values
        contribution_da = xr.DataArray(data=np.zeros((12, 264, 567)), dims=['time', 'lat', 'lon'],
                                       coords={'time': da.time, 'lat': range(264), 'lon': range(567)})
        for i_med12 in range(264):
            if i_med12 % 50 == 0:
                log_info(f"i_med = {i_med12}")
            for j_med12 in range(567):
                i = int(i_matrix[i_med12, j_med12])
                j = int(j_matrix[i_med12, j_med12])
                contribution_da[:, i_med12, j_med12] = values[:, i, j]

        weight_da = interpolation_ds[f'wgt0{index_point}']
        contribution_das.append(contribution_da * weight_da)
    # Sum contributions
    interpolated_da = sum(contribution_das)
    assert isinstance(interpolated_da, xr.DataArray)
    return interpolated_da

"""Load/save da_month_shortwave_med12"""

def get_da_raw_shortwave_med12(year: int):
    filename = get_filename(year).replace('3hr', 'month')
    filepath = ENSEMBLE_30_PATH / 'month' / 'qsr' / filename
    if filepath.exists():
        return xr.open_dataset(filepath)
    else:
        da_raw_shortwave_med12 = extract_interpolated_da(year)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        da_raw_shortwave_med12.to_netcdf(filepath)
        return da_raw_shortwave_med12

if __name__ == '__main__':
    # get_da_month_shortwave_med12()
    # get_da_raw_shortwave_med12(1980)
    for year in range(1980, 2018):
        get_da_raw_shortwave_med12(year)
