import numpy as np
import pandas as pd

from projects.ensemble_30.generate_month_nc_files import get_da_month
from projects.ensemble_30.utils_ensemble_30 import ENSEMBLE_30_PATH
from projects.scenarios.get_df_for_a_season import get_df_for_a_season
from projects.scenarios.utils_get_df_rcsm6 import compute_weights


def get_df_seasonal(ensemble_id: int, variable_name: str, extract_winter: bool) -> pd.DataFrame:
    da_month = get_da_month(variable_name, ensemble_id)
    if variable_name in ['sosstsst', 'sosaline']:
        da_month = da_month.weighted(compute_weights())
        da_month_time_series = da_month.mean(dim='x').mean(dim='y')
        if not extract_winter:
            # Remove the first month (December of the previous), when extract spring indicators
            # Otherwise it creates a spring indicator equal to Nan for the previous year
            da_month_time_series = da_month_time_series[1:]
    elif variable_name == 'qsr':
        da_month_time_series = da_month.mean(dim='x').mean(dim='y')
    else:
        raise ValueError(f'variable_name={variable_name}')
    df = get_df_for_a_season(da_month_time_series, extract_winter, variable_name)
    assert len(df) == 37, len(df)
    return df

def get_new_column_name(column_name: str):
    variable_name, extract_winter = column_name.split('_')
    variable_to_name = {
        'sosstsst': "SST",
        'sosaline': "SSS",
        'qsr': "Shortwave",
    }
    extract_winter_to_season_name = {
        'True': 'DJF',
        'False': 'MAM',
    }
    return f'{variable_to_name[variable_name]}_{extract_winter_to_season_name[extract_winter]}'


def _get_df(ensemble_id: int):
    print(f"Run _get_df for ensemble_id={ensemble_id}")
    variable_name_and_extract_winter = [('sosstsst', True), ('sosstsst', False), ('sosaline', False)]
    variable_name_and_extract_winter += [('qsr', True)]
    df_list = [get_df_seasonal(ensemble_id, variable_name, extract_winter)
               for (variable_name, extract_winter) in variable_name_and_extract_winter]
    df = pd.concat(df_list, axis=1)
    # Convert to the correct unit
    for i, column_name in enumerate(df.columns):
        if column_name.startswith('sosstsst'):
            df.iloc[:, i] += 273.15
    # Rename columns
    df.rename(columns={column_name: get_new_column_name(column_name) for column_name in df.columns}, inplace=True)
    # Add NPP columns
    df['NPP'] = 106 + 6.2 * df['SSS_MAM'] - 0.086 * df['SST_DJF'] - 1.03 * df['SST_MAM']
    df['NPP'] += 0.11 * df['Shortwave_DJF']
    return df

def get_df_for_ensemble_30(ensemble_id: int) -> pd.DataFrame:
    filepath = ENSEMBLE_30_PATH / "csv" / f"{ensemble_id}.csv"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.exists():
        df = pd.read_csv(filepath, index_col=0)
    else:
        df = _get_df(ensemble_id)
        df.to_csv(filepath)
    return df

"""Aggregate dataframes of ensemble members (minimum, average, or maximum)"""

def get_special_df_for_ensemble_30(model: str):
    if model.endswith("MEAN"):
        return get_mean_df_for_ensemble_30()
    elif model.endswith("MIN"):
        return get_min_df_for_ensemble_30()
    elif model.endswith("MAX"):
        return get_max_df_for_ensemble_30()
    else:
        raise ValueError(f'model={model}')

def get_df_list_for_ensemble_30() -> list[pd.DataFrame]:
    return [get_df_for_ensemble_30(ensemble_id) for ensemble_id in range(1, 31)]

def get_mean_df_for_ensemble_30() -> pd.DataFrame:
    df_list = get_df_list_for_ensemble_30()
    df_average = sum(df_list) / len(df_list)
    assert isinstance(df_average, pd.DataFrame)
    return df_average

def get_min_df_for_ensemble_30() -> pd.DataFrame:
    df_list = get_df_list_for_ensemble_30()
    df_min = df_list[0]
    for other_df in df_list[1:]:
        df_min = np.minimum(df_min, other_df)
    assert isinstance(df_min, pd.DataFrame)
    return df_min

def get_max_df_for_ensemble_30() -> pd.DataFrame:
    df_list = get_df_list_for_ensemble_30()
    df_max = df_list[0]
    for other_df in df_list[1:]:
        df_max = np.maximum(df_max, other_df)
    assert isinstance(df_max, pd.DataFrame)
    return df_max


if __name__ == '__main__':
    for ensemble_id in list(range(1, 31))[:1]:
        df = get_df_for_ensemble_30(ensemble_id)
        print(df.head())
