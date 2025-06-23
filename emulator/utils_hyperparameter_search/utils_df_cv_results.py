import numpy as np
import pandas as pd

from data.utils_dataset.utils_validation import get_X_and_y
from emulator.emulator import Emulator
from emulator.utils_hyperparameter_search.utils_column_names import  PARAMS_EMULATOR_COLUMN_NAME, COLUMN_NAMES
from plot.utils_metric.metric import Metric


def compute_df_cv_results(cv_results: dict, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool]) -> pd.DataFrame:
    # Pop estimator columns from cv_results dict
    emulators: list[Emulator] = cv_results.pop('estimator')
    assert all([isinstance(emulator, Emulator) for emulator in emulators])
    # Load Dataframe from cv_results
    df_cv_results = pd.DataFrame(cv_results)
    # Add params emulator
    params_emulator_list = [emulator.get_params().copy() for emulator in emulators]
    df_cv_results[PARAMS_EMULATOR_COLUMN_NAME] = params_emulator_list
    #  Add columns for the selected equation
    data = [get_series(emulator, X, y, validation_mask) for emulator in emulators]
    df_cv_results = pd.concat([df_cv_results, pd.DataFrame(index=df_cv_results.index, data=data)], axis=1)
    return df_cv_results


def get_series(emulator: Emulator, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool]) -> pd.Series:
    """For each model selection, compute RMSE train, RMSE validation, selected complexity/expr/variables names"""
    # Compute data
    rmse_train = compute_loss_for_set(emulator, X, y, validation_mask, False, Metric.RMSE)
    rmse_validation = compute_loss_for_set(emulator, X, y, validation_mask, True, Metric.RMSE)
    data = [rmse_train, rmse_validation, emulator.selected_complexity, emulator.selected_expr,
            emulator.selected_variable_names]
    # Return Series
    return pd.Series(data=data, index=COLUMN_NAMES)

def compute_loss_for_set(emulator: Emulator, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool],
                         validation_set: bool, metric: Metric) -> float:
    return emulator.compute_loss(*get_X_and_y(X, y, validation_mask, validation_set), metric=metric)


