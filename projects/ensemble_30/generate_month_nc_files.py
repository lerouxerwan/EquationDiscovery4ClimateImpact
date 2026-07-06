from pathlib import Path

import xarray as xr

from projects.ensemble_30.download_raw_nc_files import get_raw_nc_filepath
from utils.utils_path import DATA_PATH


def get_month_nc_files_directory(variable_name: str) -> Path:
    return Path(DATA_PATH) / '30_members_ensemble' / 'month' / variable_name

def get_month_filename(variable_name: str, ensemble_id: int) -> str:
    return f'{"{:03d}".format(ensemble_id)}ENS04_1m_19801201_20170501_{variable_name}.nc'

def get_month_nc_filepath(variable: str, ensemble_id: int) -> Path:
    return get_month_nc_files_directory(variable) / get_month_filename(variable, ensemble_id)

def create_month_nc_files(variable: str, ensemble_id: int):
    month_nc_filepath = get_month_nc_filepath(variable, ensemble_id)
    if month_nc_filepath.exists():
        print(f'Month data array already saved for variable={variable} and ensemble_id={ensemble_id}')
    else:
        print(f'Extract month data array for variable={variable} and ensemble_id={ensemble_id}')
        # Load raw data array
        raw_nc_filepath = get_raw_nc_filepath(variable, ensemble_id)
        ds = xr.open_dataset(raw_nc_filepath)
        # Resample to a month frequency
        month_da = ds[variable].resample(time_counter='1MS').mean(dim='time_counter')
        month_da = month_da[18:-7 ,:, :]
        # Save month data array
        month_nc_filepath.parent.mkdir(parents=True, exist_ok=True)
        month_da.to_netcdf(month_nc_filepath)

def main():
    for ensemble_id in list(range(1, 31)):
    # for ensemble_id in [30]:
        for variable_name in ['sosstsst', 'sosaline'][:]:
            create_month_nc_files(variable_name, ensemble_id)

if __name__ == '__main__':
    main()
