from typing import Optional

import numpy as np
import pandas as pd

from emulator.climate_impact_emulator import ClimateImpactEmulator
from emulator_with_search.climate_impact_emulator_with_search import ClimateImpactEmulatorWithSearch
from emulator_with_search.utils_attributes.utils_validation import get_X_and_y
from emulator.utils_plots.plot_by_split.utils_plot_split_name import SPLIT_NAMES
from utils.utils_plot import compute_axis_lim


def load_split_name_to_X_and_y(emulator: ClimateImpactEmulator, X_train: np.ndarray | pd.DataFrame,
                               y_train: np.ndarray | pd.Series,
                               X_test: Optional[np.ndarray | pd.DataFrame]=None,
                               y_test: Optional[np.ndarray | pd.Series]=None,
                               years_train: Optional[np.ndarray]=None, years_test: Optional[np.ndarray]=None) \
        -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Returns a dictionary that maps each split_name to a tuple (X,y)"""
    split_name_to_X_and_y_and_y_predict_years = load_split_name_to_X_and_y_and_y_predicted_and_years(emulator, X_train, y_train, X_test, y_test, years_train, years_test)
    return {split_name: (X, y) for split_name, (X, y, _, _) in split_name_to_X_and_y_and_y_predict_years.items()}

def load_split_name_to_X_and_y_and_y_predicted_and_years(emulator: ClimateImpactEmulator, X_train: np.ndarray | pd.DataFrame,
                                                         y_train: np.ndarray | pd.Series,
                                                         X_test: Optional[np.ndarray | pd.DataFrame]=None,
                                                         y_test: Optional[np.ndarray | pd.Series]=None,
                                                         years_train: Optional[np.ndarray]=None, years_test: Optional[np.ndarray]=None) \
        -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Returns a dictionary that maps each split_name to a tuple (X,y,years)"""
    # Some checks
    assert X_train.ndim == 2
    assert y_train.ndim == 1
    assert (years_train is None) or (years_train.ndim == 1)
    # Cast all X and y as ndarray (instead of Dataframe and Series) if it is not already done
    if isinstance(X_train, pd.DataFrame):
        X_train, y_train = X_train.values, y_train.values
        if X_test is not None:
            X_test, y_test = X_test.values, y_test.values
    # Set default for years_train and years_test if needed
    years_test, years_train = set_default_years(y_test, y_train, years_test, years_train)
    # Three splits for ClimateImpactEmulatorWithSearch
    if isinstance(emulator, ClimateImpactEmulatorWithSearch):
        #  Separate train data between train (train_train) and validation (train_validation) data
        X_train_train, y_train_train = get_X_and_y(X_train, y_train, emulator.ind_validation_, False)
        X_train_validation, y_train_validation = get_X_and_y(X_train, y_train, emulator.ind_validation_, True)
        #  Prepare ordered list with split_name, X, y and years
        split_names = SPLIT_NAMES
        X_list = [X_train_train, X_train_validation, X_test]
        y_list = [y_train_train, y_train_validation, y_test]
        years_list = [years_train[~emulator.ind_validation_], years_train[emulator.ind_validation_], years_test]
    # Two splits for ClimateImpactEmulator
    elif isinstance(emulator, ClimateImpactEmulator):
        split_names = SPLIT_NAMES[::2]
        X_list = [X_train, X_test]
        y_list = [y_train, y_test]
        years_list = [years_train, years_test]
    else:
        raise NotImplementedError(f'The class "{type(emulator)}" is not handled')
    # Remove test split (the last split name) using "zip" below if the data for this split has not been specified
    if X_test is None:
        split_names = split_names[:-1]
    # Return dictionary with split values
    split_name_to_X_and_y_and_y_predicted_and_years = dict()
    for split_name, X, y, years in zip(split_names, X_list, y_list, years_list):
        assert len(X) == len(y) == len(years)
        split_name_to_X_and_y_and_y_predicted_and_years[split_name] = (X, y, emulator.predict(X), years)
    return split_name_to_X_and_y_and_y_predicted_and_years


def set_default_years(y_test, y_train, years_test, years_train):
    if years_train is None:
        years_train = np.array(list(range(len(y_train))))
    if years_test is None:
        if y_test is not None:
            years_test = np.array(list(range(len(y_test))))
    return years_test, years_train

def get_ymin_and_ymax(split_name_to_X_and_y_and_y_predicted_and_years: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> tuple[float, float]:
    y_arrays = []
    for _, (_, y, y_predicted, _) in split_name_to_X_and_y_and_y_predicted_and_years.items():
        y_arrays.extend([y, y_predicted])
    y_values = np.concat(y_arrays)
    return compute_axis_lim(y_values)


