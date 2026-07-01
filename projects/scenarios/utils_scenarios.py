from pathlib import Path

import numpy as np
import pandas as pd

from plot.by_rcp.utils_rcp import get_rcp_label
from plot.by_rcp.utils_ssp import get_ssp_label
from projects.scenarios.get_df_rcsm4 import get_df_rcsm4
from projects.scenarios.get_df_rcsm6 import get_df_rcsm6
from utils.utils_path import DATA_PATH


def get_scenario_label(scenario: str) -> str:
    return get_rcp_label(scenario) if scenario.startswith('RCP') else get_ssp_label(scenario)

def get_marker(model: str) -> str:
    if model == 'RCSM4':
        return 'o'
    elif model == 'RCSM6':
        return 'v'
    elif model == 'RCSM6B':
        return '^'
    else:
         raise NotImplementedError

def get_color_universal(scenario: str):

    scenario_to_color = {
        'RCP45': 'tab:purple',
        'RCP85': 'r',

        'SSP370': 'gold',
        'SSP585': 'darkred',
    }
    return scenario_to_color[scenario]

def get_color(model: str, scenario: str) -> str:
    model_and_scenario_to_color = {
        ('RCSM4', 'RCP45'): 'tab:purple',
        ('RCSM4', 'RCP85'): 'r',
        ('RCSM6B', 'SSP370'): 'gold',
        ('RCSM6B', 'SSP585'): 'darkred',
        ('RCSM6', 'SSP585'): 'darkred',
    }
    return model_and_scenario_to_color[(model, scenario)]


def _get_df(model: str, scenario: str) -> pd.DataFrame:
    if model == 'RCSM4':
        return get_df_rcsm4(scenario)
    elif model == 'RCSM6':
        return get_df_rcsm6(model, scenario)
    elif model == 'RCSM6B':
        return get_df_rcsm6(model, scenario)
    else:
        raise ValueError(f"Model {model} not supported")

def get_df(model: str, scenario: str) -> pd.DataFrame:
    filepath = Path(DATA_PATH) / model / scenario / "cache.csv"
    if filepath.exists():
        df = pd.read_csv(filepath, index_col=0)
    else:
        df = _get_df(model, scenario)
        df.to_csv(filepath)
    return df


def get_years_and_values(model: str, scenario: str, variable_name: str) -> tuple[np.ndarray, np.ndarray]:
    df = get_df(model, scenario)
    years = df.index.values
    values = np.array(df[variable_name].values)
    return years, values

if __name__ == '__main__':
    df = get_df('RCSM6B', 'SSP370')
    print(df.head())