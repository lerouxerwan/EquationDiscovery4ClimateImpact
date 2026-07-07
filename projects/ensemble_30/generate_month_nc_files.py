from pathlib import Path

import xarray as xr

from projects.ensemble_30.download_raw_nc_files import get_raw_da
from projects.ensemble_30.download_raw_shortwave_nc_files import get_raw_shortwave_da
from projects.ensemble_30.utils_ensemble_30 import ENSEMBLE_30_PATH

def get_da_shortwave_month():
    years = list(range(1980, 2017))
    da_list = [get_raw_shortwave_da(year) for year in years]
    raise NotImplementedError

def get_da_month(variable_name: str, ensemble_id: int) -> xr.DataArray:
    if variable_name in ['sosstss', 'sosaline']:
        da_month = xr.open_dataset(get_month_nc_filepath(variable_name, ensemble_id))[variable_name]
        return da_month.rename({'time_counter': 'time'})
    elif variable_name == 'rsntds':
        raise get_da_shortwave_month()
    else:
        raise ValueError(f'variable_name={variable_name}')

def get_month_nc_files_directory(variable_name: str) -> Path:
    return ENSEMBLE_30_PATH / 'month' / variable_name

def get_month_filename(variable_name: str, ensemble_id: int) -> str:
    return f'{"{:03d}".format(ensemble_id)}ENS04_1m_19801201_20170501_{variable_name}.nc'

def get_month_nc_filepath(variable_name: str, ensemble_id: int) -> Path:
    return get_month_nc_files_directory(variable_name) / get_month_filename(variable_name, ensemble_id)

def create_month_nc_files(variable_name: str, ensemble_id: int):
    month_nc_filepath = get_month_nc_filepath(variable_name, ensemble_id)
    if month_nc_filepath.exists():
        print(f'Month data array already saved for variable={variable_name} and ensemble_id={ensemble_id}')
    else:
        print(f'Extract month data array for variable={variable_name} and ensemble_id={ensemble_id}')
        # Load raw data array
        raw_da = get_raw_da(variable_name, ensemble_id)
        # Resample to a month frequency
        month_da = raw_da.resample(time_counter='1MS').mean(dim='time_counter')
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
    # main()
    da = get_da_shortwave_month()
    print(da)
