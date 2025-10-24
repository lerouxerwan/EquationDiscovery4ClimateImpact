import numpy as np
import pandas as pd

from emulator.emulator import Emulator
from emulator.utils_hyperparameter_search.utils_column_names import PARAMS_EMULATOR_COLUMN_NAME, COLUMN_NAMES, \
    RMSE_TRAIN_COLUMN_NAME, RMSE_VALIDATION_COLUMN_NAME


def compute_df_cv_results(cv_results: dict, emulators: list[Emulator]) -> pd.DataFrame:
    # Load Dataframe from cv_results
    df_cv_results = pd.DataFrame(cv_results)
    # Add params emulator
    params_emulator_list = [emulator.get_params().copy() for emulator in emulators]
    df_cv_results[PARAMS_EMULATOR_COLUMN_NAME] = params_emulator_list
    #  Change the sign for the RMSE column
    for column_name in [RMSE_TRAIN_COLUMN_NAME, RMSE_VALIDATION_COLUMN_NAME]:
        df_cv_results[column_name] *= -1
    #  Add columns for the selected equation
    data = [get_series(emulator) for emulator in emulators]
    df_cv_results = pd.concat([df_cv_results, pd.DataFrame(index=df_cv_results.index, data=data)], axis=1)
    return df_cv_results


def get_series(emulator: Emulator) -> pd.Series:
    """For each model selection, compute RMSE train, RMSE validation, selected complexity/expr/variables names"""
    # Check for empty data
    if hasattr(emulator, 'equations'):
        empty_data = (emulator.equations_ is None) and (emulator.run_ is not None)
    else:
        empty_data = True
    #  Format data
    if empty_data:
        data = [np.inf, np.nan, []]
    else:
        data = [emulator.selected_complexity, emulator.selected_equation, emulator.selected_variable_names]
    # Return Series
    return pd.Series(data=data, index=COLUMN_NAMES)



