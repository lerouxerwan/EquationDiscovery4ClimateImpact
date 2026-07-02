from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pandas import DataFrame
from xarray import DataArray

from projects.scenarios.get_df_for_a_season import get_df_for_a_season
from projects.scenarios.utils_get_df_rcsm6 import compute_weights
from utils.utils_path import DATA_PATH


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
    # Extract time series list for the current scenario
    sorted_variable_files = get_sorted_variable_files(model, scenario, variable)
    time_series_list = [get_time_series(variable_file, variable, weights) for variable_file in sorted_variable_files]
    # For scenarios that are not historical, we must add the last month of December from the historical
    if scenario != 'HIST':
        last_variable_file = get_sorted_variable_files(model, 'HIST', variable)[-1]
        last_time_series = get_time_series(last_variable_file, variable, weights)
        assert last_time_series.interval_write == "1 month"
        last_time_series = last_time_series[-1:]
        datetime_for_last_time_series = to_datetime(last_time_series[-1:].time.values[0])
        assert datetime_for_last_time_series.year == 2014
        assert datetime_for_last_time_series.month == 12
        time_series_list = [last_time_series] + time_series_list
    #  Concatenate time series together
    da = xr.concat(time_series_list, dim='time')
    df = get_df_for_a_season(da, extract_winter)
    df.rename(columns={variable: f'{variable}_{extract_winter}'}, inplace=True)
    df.index.name = 'year'
    return df

def to_datetime(date):
    """Converts a numpy datetime64 object to a python datetime object"""
    timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                 / np.timedelta64(1, 's'))
    return datetime.utcfromtimestamp(timestamp)

def get_sorted_variable_files(model: str, scenario: str, variable: str) -> list[str]:
    for folder_name in ['month', 'two_years', 'decade']:
        folder_path = Path(DATA_PATH) / model / scenario / folder_name
        if folder_path.exists():
            variable_files = [str(f) for f in folder_path.iterdir() if f.name.startswith(variable)]
            if len(variable_files) > 0:
                variable_file_to_datetime = {variable_file: get_datetime(variable_file)
                                             for variable_file in variable_files}
                sorted_variable_files = sorted(variable_files, key=lambda x: variable_file_to_datetime[x])
                return sorted_variable_files
    raise ValueError(f'No file available for extraction for model={model} scenario={scenario} variable={variable}')


def get_time_series_list(sorted_variable_files: list[str], variable: str, weights: DataArray) -> list[Any]:
    #  Loop to extract variable for the area of interest
    for variable_file in sorted_variable_files:
        time_series = get_time_series(variable_file, variable, weights)
        time_series_list.append(time_series.copy())
    return time_series_list


def get_time_series(variable_file: str, variable: str, weights: DataArray) -> DataArray:
    ds = xr.open_dataset(variable_file)
    da = ds[variable].weighted(weights)
    time_series = da.mean(dim='x').mean(dim='y')
    return time_series


def get_datetime(variable_file: str) -> datetime:
    filename = variable_file.split('.')[0]
    month = int(filename[-2:])
    year = int(filename[-6:-2])
    return datetime(year, month, 1)


if __name__ == '__main__':
    # df = get_df_rcsm6('RCSM6B', 'HIST')
    # df = get_df('RCSM6B', 'HIST', 'tos', True)
    # print(df.head())
    # df = get_df_rcsm6('RCSM6B', 'SSP370')
    # print(df.head())
    pass