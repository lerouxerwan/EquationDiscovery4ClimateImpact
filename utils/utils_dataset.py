import math
import os.path as op
from typing import Optional

import numpy as np
import pandas as pd
from pysr.utils import ArrayLike

from utils.utils_path import DATASET_CSV_PATH


def load_dataset(filename_dataset: str, validation_size=0.3) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray],
Optional[ArrayLike[str]], Optional[ArrayLike[str]], np.ndarray, Optional[np.ndarray], str, Optional[str], list[str], str, np.ndarray[bool]]:
    """Load dataset parameters from a csv file, with X and y as ndarrays"""
    (X_train, y_train, X_test, y_test, X_units, y_units, years_train, years_test, rcp_name_train, rcp_name_test,
     variable_names, target_label, ind_validation) = _load_dataset_dataframe(filename_dataset, validation_size)
    variable_names = X_train.columns.to_list()
    X_test_values = None if X_test is None else X_test.values
    y_test_values = None if y_test is None else y_test.values
    return (X_train.values, y_train.values, X_test_values, y_test_values, X_units, y_units,
            years_train, years_test, rcp_name_train, rcp_name_test, variable_names, target_label, ind_validation)


def _load_dataset_dataframe(filename_dataset: str, validation_size=0.3) -> tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series],
Optional[ArrayLike[str]], Optional[ArrayLike[str]], np.ndarray, Optional[np.ndarray], str, Optional[str], Optional[np.ndarray[str]], str, np.ndarray[bool]]:
    """Load dataset parameters from a csv file, with X and y as pandas Dataframe and Series"""
    df = pd.read_csv(op.join(DATASET_CSV_PATH, filename_dataset), index_col=0)
    # Remove blank space from columns
    df.rename(columns={c: c.replace(' ', '_') for c in df.columns}, inplace=True)
    # Load features units and dataframe, target units and its series
    df, X, X_units, y, y_units = load_units(df)
    years = np.array([int(i.split('_')[-1]) for i in df.index.values])

    # Identify RCP scenarios
    prefix_set =  set([i.split('_')[0] for i in df.index.values])
    assert 1 <= len(prefix_set) <= 3
    rcp_name_train = 'RCP85'
    other_rcp_scenarios = [prefix for prefix in prefix_set if prefix.startswith('RCP') and (prefix != rcp_name_train)]
    # Extract ind_test, a boolean arrays that indicate test datapoints
    #  Split features X and target y between train split and test split
    if other_rcp_scenarios:
        rcp_name_test = other_rcp_scenarios[0]
        ind_test = df.index.str.startswith(rcp_name_test)
        X_test, y_test = X.loc[ind_test, :], y.loc[ind_test]
        X_train, y_train = X.loc[~ind_test, :], y.loc[~ind_test]
        years_train = years[~ind_test]
        years_test = years[ind_test]
    else:
        rcp_name_test = None
        X_train, y_train = X, y
        X_test, y_test = None, None
        years_train = years
        years_test = None
    # Load feature names and target name
    target_name = df.columns[:1].values[0]
    target_label = f'{target_name} ({'' if y_units is None else y_units[0]})'
    variable_names = None
    # Load index to create a validation split
    ind_validation = compute_ind_validation(len(y_train), validation_size, load_nb_historical_values(df))
    return (X_train, y_train, X_test, y_test, X_units, y_units, years_train, years_test, rcp_name_train, rcp_name_test,
            variable_names, target_label, ind_validation)

def compute_ind_validation(length: int, validation_size: float, index_start_validation: int) -> np.ndarray[bool]:
    """Compute an array of boolean such that ind_validation[i] = True if the index 'i' is in the validation set
    Parameters:
        length: int, length of the full time series
        validation_size: float, proportion (between 0 and 1) of data to include in the validation split
        index_start_validation: int, first index for the validation set
    Returns:
        ind_validation: np.ndarray[bool], ind_validation[i] = True if the index 'i' is in the validation set"""
    validation_length = math.ceil(length * validation_size)
    ind_validation = np.zeros(length).astype(bool)
    index_end_validation = validation_length + index_start_validation
    assert (0 <= index_start_validation) and (index_end_validation <= length)
    ind_validation[index_start_validation:index_end_validation] = True
    return ind_validation



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
        X_units = _load_units(series_units.iloc[1:])
        y_units = _load_units(series_units.iloc[:1])
        df = df.iloc[1:, :]
        # See https://symbolicml.org/DynamicQuantities.jl/dev/units/ for a list of accepted units
    else:
        X_units, y_units = None, None
    series_y = df.iloc[:, 0].astype(float)
    df_X = df.iloc[:, 1:].astype(float)
    return df, df_X, X_units, series_y, y_units

def _load_units(series_units: pd.Series) -> list[str]:
    return [unit if isinstance(unit, str) else '' for unit in series_units.to_list()]


if __name__ == '__main__':
    filename = r"NPP_season.csv"
    res = load_dataset(filename)
    print(res[-1])