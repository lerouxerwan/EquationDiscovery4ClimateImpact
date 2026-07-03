import os
from pathlib import Path

from utils.utils_bash_call import bash_call
from utils.utils_path import DATA_PATH

WEBPATH = "https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/MEOM/DATA_NEMOMED12"

def get_nc_files_directory(variable_name: str) -> Path:
    return Path(DATA_PATH) / '30_members_ensemble' / variable_name

def get_filename(variable_name: str, ensemble_id: int) -> str:
    return f'{"{:03d}".format(ensemble_id)}ENS04_1d_19790627_20201227_{variable_name}.nc'

def get_raw_nc_filepath(variable_name: str, ensemble_id: int) -> Path:
    return get_nc_files_directory(variable_name) / get_filename(variable_name, ensemble_id)

def download_nc_files(variable_name: str, ensemble_id: int):
    filepath = os.path.join(WEBPATH, get_filename(variable_name, ensemble_id))
    bash_call(f'wget -P {get_nc_files_directory(variable_name)}/ {filepath}')


def main():
    for ensemble_id in range(1, 16):
        # for ensemble_id in range(16,31):
        for variable_name in ['sosaline']:
            download_nc_files(variable_name, ensemble_id)


if __name__ == '__main__':
    main()