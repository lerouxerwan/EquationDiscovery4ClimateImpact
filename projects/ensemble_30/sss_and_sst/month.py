from pathlib import Path

from projects.ensemble_30.sss_and_sst.raw import get_raw_da
from projects.ensemble_30.utils_ensemble_30 import ENSEMBLE_30_PATH


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



if __name__ == '__main__':
    pass
    # for ensemble_id in list(range(1, 31)):
    #     for variable_name in ['sosstsst', 'sosaline'][:]:
    #         create_month_nc_files(variable_name, ensemble_id)
