import os.path as op
from typing import Optional

import numpy as np
import pandas as pd
from pysr.utils import ArrayLike

from utils.utils_path import DATASET_CSV_PATH


def load_dataset_ndarray(filename_dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
Optional[ArrayLike[str]], Optional[ArrayLike[str]], np.ndarray, np.ndarray, str, str, np.ndarray[str], str, int]:
    """Load dataset parameters from a csv file, with X and y as ndarrays"""
    (X_train, y_train, X_test, y_test, X_units, y_units, years_train, years_test, rcp_name_train, rcp_name_test,
     variable_names, target_label, nb_historical_years) = load_dataset_dataframe(filename_dataset)
    variable_names = X_train.columns.values
    return (X_train.values, y_train.values, X_test.values, y_test.values, X_units, y_units,
            years_train, years_test, rcp_name_train, rcp_name_test, variable_names, target_label, nb_historical_years)


def load_dataset_dataframe(filename_dataset: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series,
Optional[ArrayLike[str]], Optional[ArrayLike[str]], np.ndarray, np.ndarray, str, str, Optional[np.ndarray[str]], str, int]:
    """Load dataset parameters from a csv file, with X and y as pandas Dataframe and Series"""
    df = pd.read_csv(op.join(DATASET_CSV_PATH, filename_dataset), index_col=0)
    # Remove blank space from columns
    df.rename(columns={c: c.replace(' ', '_') for c in df.columns}, inplace=True)
    # Load features units and dataframe, target units and its series
    df, X, X_units, y, y_units = load_units(df)
    # Identify the two RCP scenarios
    prefix_set =  set([i.split('_')[0] for i in df.index.values])
    assert 1 <= len(prefix_set) <= 3
    rcp_name_train = 'RCP85'
    rcp_scenarios = [prefix for prefix in prefix_set if prefix.startswith('RCP') and (prefix != rcp_name_train)]
    rcp_name_test = rcp_scenarios[0]
    # Split features X and target y between train split and test split
    ind_test = df.index.str.startswith(rcp_name_test)
    X_test, y_test = X.loc[ind_test, :], y.loc[ind_test]
    X_train, y_train = X.loc[~ind_test, :], y.loc[~ind_test]
    # Load corresponding years for the data
    years = np.array([int(i.split('_')[-1]) for i in df.index.values])
    years_train = years[~ind_test]
    years_test = years[ind_test]
    # Load feature names and target name
    target_name = df.columns[:1].values[0]
    target_label = f'{target_name} ({'' if y_units is None else y_units[0]})'
    variable_names = None
    # Load index to create a validation split
    nb_historical_years = load_nb_historical_values(df)
    return (X_train, y_train, X_test, y_test, X_units, y_units, years_train, years_test, rcp_name_train, rcp_name_test,
            variable_names, target_label, nb_historical_years)


def load_nb_historical_values(df: pd.DataFrame) -> int:
    """This index corresponds to the start of the rcp scenario, i.e. the start of the RCP scenario for training"""
    for index, (index_name, _) in enumerate(df.iterrows()):
        prefix = index_name[:3]
        if prefix == 'RCP':
            return index
        else:
            assert prefix == 'HIS'
    raise ValueError('No row of the dataframe starts with "RCP"')

def load_units(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Optional[list[str]], pd.Series, Optional[list[str]]]:
    """Extract the row 'UNIT' then remove it from df (if the row exists)"""
    if 'UNIT' in df.index:
        series_units = df.loc['UNIT']
        X_units = series_units.iloc[1:].to_list()
        y_units = series_units.iloc[:1].to_list()
        df = df.iloc[1:, :]
        # See https://symbolicml.org/DynamicQuantities.jl/dev/units/ for a list of accepted units
    else:
        X_units, y_units = None, None
    series_y = df.iloc[:, 0].astype(float)
    df_X = df.iloc[:, 1:].astype(float)
    return df, df_X, X_units, series_y, y_units


if __name__ == '__main__':
    filename = r"NPP_month.csv"
    res = load_dataset_dataframe(filename)
    print(res[-1])