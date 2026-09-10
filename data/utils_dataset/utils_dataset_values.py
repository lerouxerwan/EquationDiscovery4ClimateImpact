import os.path as op
from typing import Optional

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from data.utils_dataset.utils_validation_split import get_validation_mask
from data.utils_dataset.validation_split import ValidationSplit
from utils.utils_path import DATASET_CSV_PATH


def load_dataset_values(csv_filename: str, rcp_name_train: str, rcp_name_test: Optional[str] = None,
                        validation_size: float = 0.3, validation_split: ValidationSplit = ValidationSplit.RCP_START) \
        -> tuple[
            ndarray, ndarray, Optional[ndarray], Optional[ndarray], ndarray, Optional[ndarray],
            Optional[list[str]], Optional[list[str]], Optional[list[str]], Optional[list[str]], list[str], list[str],
            ndarray[bool], int
        ]:
    """Load dataset values from a csv filename.
        -Values for each set (train and test) including X an array, y an array, years an array of int.
        -Values for the target & features including units str list, labels str list, variable_names str list
        -Values for the validation set including validation_mask an array of boolean
    So far the code only handles data coming from two RCPs:
        -one RCP for the train set (rcp_name_train), If historical data are available, they are added to the train set
        -an optional RCP for the test set (rcp_name_test)
    Two additional rows can be in the csv file:
        -'LABEL' which defines as string a long label for plots (for instance it can contain units in latex format)
        -'UNIT' which defines with a string the standard physical unit for the target and features,
            See https://symbolicml.org/DynamicQuantities.jl/dev/units/ for a list of accepted units"""
    # Load csv file
    assert csv_filename.endswith('.csv')
    df = pd.read_csv(op.join(DATASET_CSV_PATH, csv_filename), index_col=0)
    # Extract values for the target y & features X. Drop potential additional rows ('UNIT' and 'LABEL') in the DataFrame
    df, X_units, y_units = load_additional_row(df, 'UNIT')
    df, X_labels, y_labels = load_additional_row(df, 'LABEL')
    variable_names = list(df.columns)
    X_variable_names, y_variable_names = variable_names[1:], variable_names[:1]
    # Cast dataframe to the float type
    df = df.astype(float)
    # Check dataframe: at best 2 RCP scenarios should be in the index values (and that other prefix can only be 'HIST')
    prefixes = np.array([str(i.split('_')[0]) for i in df.index.values])
    nb_historical_years = sum([prefix == 'HIST' for prefix in prefixes])
    prefix_set = set(list(prefixes))
    found_rcp_scenarios = set([prefix for prefix in prefix_set if prefix.startswith('RCP')])
    expected_rcp_scenarios = {rcp_name_train} if rcp_name_test is None else {rcp_name_train, rcp_name_test}
    assert found_rcp_scenarios == expected_rcp_scenarios
    assert (prefix_set - found_rcp_scenarios).issubset({'HIST'})
    # Load values (X, y, years) for the train set and test set
    X, y = df.iloc[:, 1:], df.iloc[:, 0]
    years = np.array([int(i.split('_')[-1]) for i in df.index.values])
    ind_test = None if rcp_name_test is None else df.index.str.startswith(rcp_name_test)
    if ind_test is None:
        X_train, X_test = X.values, None
        y_train, y_test = y.values, None
        years_train, years_test = years, None
    else:
        X_train, X_test = X.loc[~ind_test, :].values, X.loc[ind_test, :].values
        y_train, y_test = y.loc[~ind_test].values, y.loc[ind_test].values
        years_train, years_test = years[~ind_test], years[ind_test]
    # Load validation mask
    prefixes_train = prefixes if (ind_test is None) else prefixes[~ind_test]
    validation_mask = get_validation_mask(y_train, validation_size, validation_split,
                                          list(years_train), rcp_name_train, list(prefixes_train))
    return (
        X_train, y_train, X_test, y_test, years_train, years_test,
        X_units, y_units, X_labels, y_labels, X_variable_names, y_variable_names,
        validation_mask, nb_historical_years
    )

def load_additional_row(df: pd.DataFrame, additional_row_name: str) -> tuple[DataFrame, Optional[list[str]], Optional[list[str]]]:
    """Extract and drop an additional row in the csv file"""
    if additional_row_name in df.index:
        def _load_str(series: pd.Series) -> list[str]:
            return [s if isinstance(s, str) else '' for s in series.to_list()]
        series = df.loc[additional_row_name]
        X_values, y_values = _load_str(series.iloc[1:]), _load_str(series.iloc[:1])
        df.drop(additional_row_name, inplace=True)
    else:
        X_values, y_values = None, None
    return df, X_values, y_values