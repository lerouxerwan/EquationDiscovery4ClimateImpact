import xarray as xr
from datetime import datetime
from pathlib import Path

import pandas as pd

from projects.scenarios.get_df_from_decade_file import get_df_from_decade_file
from projects.scenarios.get_df_from_month_file import get_df_from_month_file
from projects.scenarios.get_df_from_two_years_file import get_df_from_two_years_file
from projects.scenarios.utils_get_df_rcsm6 import compute_weights
from utils.utils_path import DATA_PATH

folder_name_to_extraction_function = {
    'two_years': get_df_from_two_years_file,
    'month': get_df_from_two_years_file,
    'decade': get_df_from_two_years_file,
    # 'month': get_df_from_month_file,
    # 'decade': get_df_from_decade_file,
}

def get_df_rcsm6(model: str, scenario: str):
    variable_name_and_extract_winter = [('tos', True), ('tos', False), ('sos', False), ('rsntds', True)]
    df_list = [get_df(model, scenario, variable_name, extract_winter)
               for (variable_name, extract_winter) in variable_name_and_extract_winter]
    df = pd.concat(df_list, axis=1)
    # Convert to the correct unit
    for i, column_name in enumerate(df.columns):
        if column_name.startswith('tos'):
            df.iloc[:, i] += 273.15
    # Rename columns
    df.rename(columns={column_name: get_new_column_name(column_name) for column_name in df.columns}, inplace=True)
    # Add NPP columns
    df['NPP'] = 106 + 6.2 * df['SSS_MAM'] - 0.086 * df['SST_DJF'] - 1.03 * df['SST_MAM'] + 0.11 * df['Shortwave_DJF']
    return df

def get_new_column_name(column_name: str):
    variable_name, extract_winter = column_name.split('_')
    variable_to_name = {
        'tos': "SST",
        'sos': "SSS",
        'rsntds': "Shortwave",
    }
    extract_winter_to_season_name = {
        'True': 'DJF',
        'False': 'MAM',
    }
    return f'{variable_to_name[variable_name]}_{extract_winter_to_season_name[extract_winter]}'

def get_df(model: str, scenario: str, variable: str, extract_winter: bool):
    weights = compute_weights()
    for folder_name, extract_function in folder_name_to_extraction_function.items():
        folder_path = Path(DATA_PATH) / model / scenario / folder_name
        if folder_path.exists():
            variable_files = [str(f) for f in folder_path.iterdir() if f.name.startswith(variable)]
            if len(variable_files) > 0:
                variable_file_to_datetime = {variable_file:  get_datetime(variable_file)
                                             for variable_file in variable_files}
                sorted_variable_files = sorted(variable_files, key=lambda x: variable_file_to_datetime[x])
                #  Loop to extract variable for the area of interest
                time_series_list = []
                for f in sorted_variable_files:
                    ds = xr.open_dataset(f)
                    da = ds[variable].weighted(weights)
                    time_series = da.mean(dim='x').mean(dim='y')
                    time_series_list.append(time_series.copy())
                #  Concatenate time series together
                da = xr.concat(time_series_list, dim='time')
                df = extract_function(da, extract_winter)
                df.rename(columns={variable: f'{variable}_{extract_winter}'}, inplace=True)
                df.index.name = 'year'
                return df
    raise ValueError('No file available for extraction')

def get_datetime(variable_file: str) -> datetime:
    filename = variable_file.split('.')[0]
    month = int(filename[-2:])
    year = int(filename[-6:-2])
    return datetime(year, month, 1)


if __name__ == '__main__':
    df = get_df_rcsm6('RCSM6', 'HIST')
    print(df.head())
