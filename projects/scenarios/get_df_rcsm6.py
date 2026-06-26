from pathlib import Path

import pandas as pd

from projects.scenarios.get_df_from_decade_file import get_df_from_decade_file
from projects.scenarios.get_df_from_month_file import get_df_from_month_file
from projects.scenarios.get_df_from_two_years_file import get_df_from_two_years_file
from projects.scenarios.utils_get_df_rcsm6 import compute_weights
from utils.utils_path import DATA_PATH

variable_name_to_extract_winter = {
    'tos': True,
    'tos': False,
    'sos': False,
    'rsntds': True,
}

folder_name_to_extraction_function = {
    'two_years': get_df_from_two_years_file,
    'month': get_df_from_month_file,
    'decade': get_df_from_decade_file,
}

def get_df_rcsm6(model: str, scenario: str):

    df_list = [get_df(model, scenario, variable_name, extract_winter)
               for (variable_name, extract_winter) in variable_name_to_extract_winter.items()]
    df = pd.concat(df_list, axis=1)
    df['NPP'] = 106 + 6.2 * df['SSS_MAM'] - 0.086 * df['SST_DJF'] - 1.03 * df['SST_MAM'] + 0.11 * df['Shortwave_DJF']
    return df


def get_df(model: str, scenario: str, variable: str, extract_winter: bool):
    weights = compute_weights()
    for folder_name, extract_function in folder_name_to_extraction_function.items():
        folder_path = Path(DATA_PATH) / model / scenario / folder_name
        if folder_path.exists():
            variable_files = [f for f in folder_path.iterdir() if f.name.startswith(variable)]
            if len(variable_files) > 0:
                return extract_function(variable, extract_winter, weights, variable_files)
    raise ValueError('No file available for extraction')


if __name__ == '__main__':
    df = get_df('RCSM6B', 'HIST', 'sos', True)
    print(df.head())
    df = get_df('RCSM6B', 'HIST', 'tos', True)
    print(df.head())
    df = get_df_rcsm6('RCSM6B', 'HIST')
    print(df.head())
