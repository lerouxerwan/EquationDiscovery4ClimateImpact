from typing import Optional

import pandas as pd
from pysr.utils import ArrayLike
from sympy import Symbol

from emulator.pysr_emulator import PySREmulator

RANK_COLUMN_NAME = 'rank_test_MSE'
METRIC_COLUMN_NAME = 'mean_test_MSE'
RMSE_VALIDATION_COLUMN_NAME = 'RMSE_validation'
SELECTED_COMPLEXITY_COLUMN_NAME = 'selected_complexity'
SELECTED_EXPR_COLUMN_NAME = 'selected_expr'
SELECTED_FEATURE_INDEXES_COLUMN_NAME = 'selected_feature_indexes'
PARAMS_EMULATOR_COLUMN_NAME = 'params_emulator'

def get_df_cv_results(cv_results: dict, variable_names: ArrayLike[str] | None = None) -> pd.DataFrame:
    # Pop estimator columns
    emulators = cv_results.pop('estimator')
    assert all([isinstance(emulator, PySREmulator) for emulator in emulators])
    #  Remove deprecated losses
    for key in ['split0_test_MSE', 'mean_test_MSE', 'std_test_MSE', 'rank_test_MSE']:
        cv_results.pop(key)

    # Load Dataframe from cv_results
    df_cv_results = pd.DataFrame(cv_results)
    # Compute values for columns
    selected_complexity_list = [emulator.selected_complexity for emulator in emulators]
    selected_expr_list = [emulator.selected_expr for emulator in emulators]
    params_emulator_list = [emulator.get_params().copy() for emulator in emulators]
    selected_variable_names_list = [emulator.selected_variable_names for emulator in emulators]
    selected_feature_indexes = [get_selected_feature_indexes(selected_variable_names, variable_names)
                                for selected_variable_names in selected_variable_names_list]
    # Add columns to the DataFrame
    df_cv_results[SELECTED_COMPLEXITY_COLUMN_NAME] = selected_complexity_list
    df_cv_results[SELECTED_EXPR_COLUMN_NAME] = selected_expr_list
    df_cv_results[SELECTED_FEATURE_INDEXES_COLUMN_NAME] = selected_feature_indexes
    df_cv_results[PARAMS_EMULATOR_COLUMN_NAME] = params_emulator_list
    # Sort DataFrame by the rank
    df_cv_results = df_cv_results.sort_values(by=RMSE_VALIDATION_COLUMN_NAME)
    return df_cv_results

def get_selected_feature_indexes(selected_variable_names: list[str], variable_names: Optional[ArrayLike[str]] = None) -> list[int]:
    if variable_names is None:
        return [int(selected_variable_name[1:]) for selected_variable_name in selected_variable_names]
    else:
        return [variable_names.index(selected_variable_name) for selected_variable_name in selected_variable_names]


