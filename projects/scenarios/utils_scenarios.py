from pathlib import Path

import numpy as np
import pandas as pd

from plot.by_rcp.utils_rcp import get_rcp_label
from plot.by_rcp.utils_ssp import get_ssp_label
from projects.ensemble_30.df_for_ensemble_30 import get_mean_df_for_ensemble_30, get_special_df_for_ensemble_30
from projects.scenarios.get_df_rcsm4 import get_df_rcsm4
from projects.scenarios.get_df_rcsm6 import get_df_rcsm6
from utils.utils_path import DATA_PATH

START_REFERENCE_YEAR = 1986
END_REFERENCE_YEAR = 2014

def get_linewidth(model: str) -> float:
    default_linewidth = 4
    if model in ['MEOM_MIN', 'MEOM_MAX']:
        return default_linewidth / 2
    else:
        return default_linewidth


def get_scenario_label(scenario: str) -> str:
    if scenario == "ENS04":
        return "ENS04"
    else:
        return get_rcp_label(scenario) if scenario.startswith('RCP') else get_ssp_label(scenario)

def get_marker(model: str) -> str:
    if model == 'RCSM4':
        return 'o'
    elif model == 'RCSM6':
        return '<'
    elif model == 'RCSM6B':
        return '>'
    elif model == "MEOM_MEAN":
        return 'D'
    elif model == "MEOM_MIN":
        return 'v'
    elif model == "MEOM_MAX":
        return '^'
    else:
         raise NotImplementedError

def get_color(scenario: str):

    scenario_to_color = {
        'RCP45': 'tab:purple',
        'RCP85': 'r',

        'SSP370': 'gold',
        'SSP585': 'darkred',
        'ENS04': 'green',
    }
    return scenario_to_color[scenario]

def get_df(model: str, scenario: str) -> pd.DataFrame:
    if model.startswith("MEOM"):
        assert scenario in ['HIST', 'ENS04']
        df = get_special_df_for_ensemble_30(model)
        df = df.loc[:END_REFERENCE_YEAR] if scenario == 'HIST' else df.loc[END_REFERENCE_YEAR+1:]
        return df
    else:
        filepath = Path(DATA_PATH) / model / scenario / "cache.csv"
        if filepath.exists():
            df = pd.read_csv(filepath, index_col=0)
        else:
            if model == 'RCSM4':
                df = get_df_rcsm4(scenario)
            elif model in ['RCSM6', 'RCSM6B']:
                df = get_df_rcsm6(model, scenario)
            else:
                raise ValueError(f"Model {model} not supported")
            df.to_csv(filepath)
        return df


def get_years_and_values(model: str, scenario: str, variable_name: str) -> tuple[np.ndarray, np.ndarray]:
    df = get_df(model, scenario)
    if variable_name not in df.columns:
        assert variable_name in ['NPP_without_shortwave_term', 'NPP_from_shortwave_term', 'relative_contribution_NPP_for_shortwave_term']
        df['NPP_without_shortwave_term'] = 106 + 6.2 * df['SSS_MAM'] - 0.086 * df['SST_DJF'] - 1.03 * df['SST_MAM']
        df['NPP_from_shortwave_term'] = df['NPP'] - df['NPP_without_shortwave_term']
        df['NPP'] = 106 + 6.2 * df['SSS_MAM'] - 0.086 * df['SST_DJF'] - 1.03 * df['SST_MAM'] + 0.11 * df['Shortwave_DJF']
        df['relative_contribution_NPP_for_shortwave_term'] = 100 * (0.11 * df['Shortwave_DJF'].abs())
        df['relative_contribution_NPP_for_shortwave_term'] /= (106 + 6.2 * df['SSS_MAM'].abs() + 0.086 * df['SST_DJF'] + 1.03 * df['SST_MAM'] + 0.11 * df['Shortwave_DJF'].abs())
    return df.index.values, np.array(df[variable_name].values)

if __name__ == '__main__':
    df = get_df('RCSM6B', 'SSP370')
    print(df.head())