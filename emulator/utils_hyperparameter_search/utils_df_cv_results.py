import numpy as np
import pandas as pd

from emulator.emulator import Emulator
from emulator.utils_hyperparameter_search.utils_column_names import get_cv_results_column_names, \
    PARAMS_EMULATOR_COLUMN_NAME
from plot.utils_metric.metric import Metric


def compute_df_cv_results(cv_results: dict, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool]) -> pd.DataFrame:
    # Pop estimator columns from cv_results dict
    emulators = cv_results.pop('estimator')
    assert all([isinstance(emulator, Emulator) for emulator in emulators])
    # Load Dataframe from cv_results
    df_cv_results = pd.DataFrame(cv_results)
    # Add params emulator
    params_emulator_list = [emulator.get_params().copy() for emulator in emulators]
    df_cv_results[PARAMS_EMULATOR_COLUMN_NAME] = params_emulator_list
    #  Add columns for the selected equations for each model_selection
    for model_selection in ['best', 'validated']:
        data = [get_series(model_selection, emulator, X, y, validation_mask) for emulator in emulators]
        df_model_selection = pd.DataFrame(index=df_cv_results.index, data=data)
        df_cv_results = pd.concat([df_cv_results, df_model_selection], axis=1)
    return df_cv_results


def get_series(model_selection: str, emulator: Emulator, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool]) -> pd.Series:
    """Get the added series, a row with new column that will be appended to the initial cv_results"""
    # Save original attributes
    original_threshold = emulator.threshold_for_model_selection
    original_model_selection = emulator.model_selection[:] # copy model_selection
    # Compute the row
    emulator.model_selection = model_selection
    if model_selection == 'best':
        emulator.threshold_for_model_selection = 1.5
    elif model_selection == 'validated':
        emulator.set_threshold_for_model_selection_validated(X, y, validation_mask)
    else:
        raise NotImplementedError
    series = _get_series(model_selection, emulator, X, y, validation_mask)
    # Reset to original attributes
    emulator.threshold_for_model_selection = original_threshold
    emulator.model_selection = original_model_selection
    return series

def _get_series(model_selection: str, emulator: Emulator, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool]) -> pd.Series:
    """For each model selection, compute RMSE train, RMSE validation, selected complexity/expr/variables names"""
    # Compute data
    rmse_train = emulator.compute_loss_for_set(X, y, validation_mask, False, Metric.RMSE)
    rmse_validation = emulator.compute_loss_for_set(X, y, validation_mask, True, Metric.RMSE)
    data = [rmse_train, rmse_validation, emulator.selected_complexity, emulator.selected_expr,
            emulator.selected_variable_names]
    # Return Series
    return pd.Series(data=data, index=get_cv_results_column_names(model_selection))

