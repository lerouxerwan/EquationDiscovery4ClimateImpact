import xarray as xr
from pathlib import Path

from projects.ensemble_30.download_nc_files import get_nc_files_directory, get_filename, get_raw_nc_filepath


def get_monthly_nc_files_directory(variable_name: str) -> Path:
    return Path(str(get_nc_files_directory(variable_name)).replace('raw', 'csv'))

def get_monthly_nc_filepath(variable_name, ensemble_id) -> Path:
    return get_monthly_nc_files_directory(variable_name) / get_filename(variable_name, ensemble_id)

def create_monthly_nc_files(variable_name: str, ensemble_id: int):
    raw_nc_filepath = get_raw_nc_filepath(variable_name, ensemble_id)
    raw_da = xr.open_dataset(raw_nc_filepath)

    monthly_da = raw_da.resample(time_counter='1M').mean()
    monthly_nc_filepath = get_monthly_nc_filepath(variable_name, ensemble_id)




def main():
    for ensemble_id in range(1, 16):
        # for ensemble_id in range(16,31):
        for variable_name in ['sosaline']:
            create_monthly_nc_files(variable_name, ensemble_id)

if __name__ == '__main__':
    main()
