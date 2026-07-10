import os
from pathlib import Path

import xarray as xr

from projects.ensemble_30.sss_and_sst.raw import get_raw_nc_files_directory
from projects.ensemble_30.utils_ensemble_30 import WEBPATH
from utils.utils_bash_call import bash_call


def get_filename(year: int) -> str:
    return f"qsr_MED-11_ECMWF-ERA5_evaluation_r1i1p1_CNRM-ALADIN64_v1_3hr_{year}.nc"

def get_da(year: int) -> xr.DataArray:
    filepath = get_raw_nc_files_directory("qsr") / get_filename(year)
    return xr.open_dataset(filepath)["qsr"]

def download(year: int):
    filepath = os.path.join(WEBPATH, get_filename(year))
    bash_call(f'wget -P {get_raw_nc_files_directory("qsr")}/ {filepath}')


def main():
    for year in list(range(1980, 2018))[-1:]:
        download(year)

if __name__ == '__main__':
    main()