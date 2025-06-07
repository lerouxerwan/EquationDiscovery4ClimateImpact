from typing import Optional

import numpy as np
import pandas as pd
from pysr.utils import ArrayLike

from emulator.emulator_validated import EmulatorValidated
from emulator.utils_hyperparameter_search.utils_column_names import COLUMN_NAMES, get_cv_results_column_names
from plot.utils_metric.metric import Metric


def get_series(model_selection: str, emulator: EmulatorValidated, X: np.ndarray, y: np.ndarray,
               validation_mask: np.ndarray[bool], variable_names: ArrayLike[str] | None) -> pd.Series:
    """Get the added series, a row with new column that will be appended to the initial cv_results"""
    # Save original attributes
    original_threshold = emulator.threshold_for_model_selection
    original_model_selection = emulator.model_selection[:] # copy model_selection
    # Compute the row
    emulator.model_selection = model_selection
    if model_selection == 'best':
        emulator.threshold_for_model_selection = 1.5
    elif model_selection == 'custom':
        emulator.set_threshold_for_custom_model_selection(X, y, validation_mask)
    else:
        raise NotImplementedError
    series = _get_series(model_selection, emulator, X, y, validation_mask, variable_names)
    # Reset to original attributes
    emulator.threshold_for_model_selection = original_threshold
    emulator.model_selection = original_model_selection
    return series

def _get_series(model_selection: str, emulator: EmulatorValidated, X: np.ndarray, y: np.ndarray,
               validation_mask: np.ndarray[bool], variable_names: ArrayLike[str] | None) -> pd.Series:
    """For each model selection, compute RMSE train, RMSE val, selected complexity/expr/features/variables names"""
    # Compute data
    rmse_train = emulator.compute_loss_for_train_or_validation_set(X, y, validation_mask, False, Metric.RMSE)
    rmse_validation = emulator.compute_loss_for_train_or_validation_set(X, y, validation_mask, True, Metric.RMSE)
    data = [rmse_train, rmse_validation, emulator.selected_complexity, emulator.selected_expr,
            emulator.selected_variable_names, get_selected_feature_indexes(emulator.selected_variable_names, variable_names)]
    # Return Series
    return pd.Series(data=data, index=get_cv_results_column_names(model_selection))

def get_selected_feature_indexes(selected_variable_names: list[str], variable_names: Optional[ArrayLike[str]] = None) -> list[int]:
    if variable_names is None:
        return [int(selected_variable_name[1:]) for selected_variable_name in selected_variable_names]
    else:
        return [variable_names.index(selected_variable_name) for selected_variable_name in selected_variable_names]


