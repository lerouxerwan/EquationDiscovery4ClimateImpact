import math
import os.path as op
from typing import Optional

import numpy as np
import pandas as pd
from pysr.utils import ArrayLike

from data.utils_dataset.utils_units import load_units
from data.utils_dataset.utils_validation import compute_validation_mask
from utils.utils_path import DATASET_CSV_PATH


def load_dataset(filename_dataset: str, validation_size=0.3) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray],
Optional[ArrayLike[str]], Optional[ArrayLike[str]], np.ndarray, Optional[np.ndarray], str, Optional[str], list[str], str, np.ndarray[bool]]:
    """Load dataset parameters from a csv file, with X and y as ndarrays"""
    (X_train, y_train, X_test, y_test, X_units, y_units, years_train, years_test, rcp_name_train, rcp_name_test,
     variable_names, target_label, validation_mask) = _load_dataset_dataframe(filename_dataset, validation_size)
    variable_names = X_train.columns.to_list()
    X_test_values = None if X_test is None else X_test.values
    y_test_values = None if y_test is None else y_test.values
    return (X_train.values, y_train.values, X_test_values, y_test_values, X_units, y_units,
            years_train, years_test, rcp_name_train, rcp_name_test, variable_names, target_label, validation_mask)


def _load_dataset_dataframe(filename_dataset: str, validation_size=0.3) -> tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series],
Optional[ArrayLike[str]], Optional[ArrayLike[str]], np.ndarray, Optional[np.ndarray], str, Optional[str], Optional[np.ndarray[str]], str, np.ndarray[bool]]:
    """Load dataset parameters from a csv file, with X and y as pandas Dataframe and Series"""
    df = pd.read_csv(op.join(DATASET_CSV_PATH, filename_dataset), index_col=0)
    # Remove blank space from columns
    df.rename(columns={c: '_'.join(c.split()[1:]) for c in df.columns[1:]}, inplace=True)
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
    target_label = target_label.replace('g / yr', "gC $\;$ year$^{-1}$")
    variable_names = None
    # Load index to create a validation split
    validation_mask = compute_validation_mask(y_train, validation_size, df)
    return (X_train, y_train, X_test, y_test, X_units, y_units, years_train, years_test, rcp_name_train, rcp_name_test,
            variable_names, target_label, validation_mask)


if __name__ == '__main__':
    filename = r"NPP_season.csv"
    res = load_dataset(filename)
    print(res[-1])