import os
from pathlib import Path

import xarray as xr
from matplotlib import pyplot as plt

from projects.ensemble_30.download_raw_nc_files import get_raw_nc_files_directory
from projects.ensemble_30.utils_ensemble_30 import WEBPATH
from utils.utils_bash_call import bash_call

VARIABLE_NAME = "rsntds"


def get_raw_shortwave_filename(year: int) -> str:
    return f"qsr_MED-11_ECMWF-ERA5_evaluation_r1i1p1_CNRM-ALADIN64_v1_3hr_{year}.nc"
    return f'{"{:03d}".format(ensemble_id)}ENS04_1d_19790627_20201227_{variable_name}.nc'

def get_raw_shortwave_nc_filepath(year: int) -> Path:
    return get_raw_nc_files_directory(VARIABLE_NAME) / get_raw_shortwave_filename(year)

def get_raw_shortwave_da(year: int) -> xr.DataArray:
    ds = xr.open_dataset(get_raw_shortwave_nc_filepath(year))
    for t in [150, 1000, 2000]:
        ds.qsr[t, :, :].plot()
        plt.show()
    return ds["variable_name"]

def download_raw_shortwave_nc_files(year: int):
    filepath = os.path.join(WEBPATH, get_raw_shortwave_filename(year))
    bash_call(f'wget -P {get_raw_nc_files_directory(VARIABLE_NAME)}/ {filepath}')


def main():
    for year in list(range(1980, 2017))[:]:
        download_raw_shortwave_nc_files(year)


if __name__ == '__main__':
    main()