import numpy as np
import pandas as pd

from emulator.pysr_emulator import PySREmulator

RANK_COLUMN_NAME = 'rank_test_MSE'
METRIC_COLUMN_NAME = 'mean_test_MSE'
RMSE_VALIDATION_COLUMN_NAME = 'RMSE_validation'
SELECTED_COMPLEXITY_COLUMN_NAME = 'selected_complexity'
SELECTED_EXPR_COLUMN_NAME = 'selected_expr'
PARAMS_EMULATOR_COLUMN_NAME = 'params_emulator'

def get_df_cv_results(cv_results: dict) -> pd.DataFrame:
    # Pop estimator columns
    emulators = cv_results.pop('estimator')
    assert all([isinstance(emulator, PySREmulator) for emulator in emulators])
    #  Remove deprecated losses
    for key in ['split0_test_MSE', 'mean_test_MSE', 'std_test_MSE', 'rank_test_MSE']:
        cv_results.pop(key)

    # Load Dataframe from cv_results
    df_cv_results = pd.DataFrame(cv_results)
    # Add columns to the DataFrame
    df_cv_results[SELECTED_COMPLEXITY_COLUMN_NAME] = [emulator.selected_complexity for emulator in emulators]
    df_cv_results[SELECTED_EXPR_COLUMN_NAME] = [emulator.selected_expr for emulator in emulators]
    df_cv_results[PARAMS_EMULATOR_COLUMN_NAME] = [emulator.get_params().copy() for emulator in emulators]
    # Sort DataFrame by the rank
    df_cv_results = df_cv_results.sort_values(by=RMSE_VALIDATION_COLUMN_NAME)
    return df_cv_results

