import os
from pathlib import Path

import xarray as xr

from projects.ensemble_30.utils_ensemble_30 import ENSEMBLE_30_PATH, RAW_PATH, WEBPATH
from utils.utils_bash_call import bash_call



def get_raw_nc_files_directory(variable_name: str) -> Path:
    # return Path(r'/media/e23lerou/LaCie/Mediation/30_members_ensemble/raw')
    return RAW_PATH / variable_name

def get_raw_filename(variable_name: str, ensemble_id: int) -> str:
    return f'{"{:03d}".format(ensemble_id)}ENS04_1d_19790627_20201227_{variable_name}.nc'

def get_raw_da(variable_name: str, ensemble_id: int) -> xr.DataArray:
    filepath = get_raw_nc_files_directory(variable_name) / get_raw_filename(variable_name, ensemble_id)
    return xr.open_dataset(filepath)[variable_name]

def download(variable_name: str, ensemble_id: int):
    filepath = os.path.join(WEBPATH, get_raw_filename(variable_name, ensemble_id))
    bash_call(f'wget -P {get_raw_nc_files_directory(variable_name)}/ {filepath}')


def main():
    # for ensemble_id in range(1, 16):
    for ensemble_id in range(16,31):
        for variable_name in ['sosstsst', 'sosaline'][:1]:
            download(variable_name, ensemble_id)


if __name__ == '__main__':
    main()