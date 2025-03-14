import numpy as np
import pandas as pd

from emulator.pysr_emulator import PySREmulator
from emulator.utils_cache.utils_key import get_key_for_cache_fit

RANK_COLUMN_NAME = 'rank_test_MSE'
METRIC_COLUMN_NAME = 'mean_test_MSE'
RMSE_VALIDATION_COLUMN_NAME = 'RMSE_validation'
SELECTED_COMPLEXITY_COLUMN_NAME = 'selected_complexity'
SELECTED_EXPR_COLUMN_NAME = 'selected_expr'
PARAMS_EMULATOR_COLUMN_NAME = 'params_emulator'

def get_df_cv_results(cv_results: dict, emulator: PySREmulator, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    # Load Dataframe from cv_results
    df_cv_results = pd.DataFrame(cv_results).sort_values(by=RANK_COLUMN_NAME)
    assert df_cv_results[RANK_COLUMN_NAME].values[0] == 1
    # Add column with the RMSE Validation
    df_cv_results[RMSE_VALIDATION_COLUMN_NAME] = df_cv_results[METRIC_COLUMN_NAME].apply(lambda x: np.sqrt(-x))
    # Extract list of selected_complexity, selected_expression and params_emulator
    line_values = []
    for line, params in enumerate(df_cv_results["params"].values, 1):
        params_emulator = emulator.set_params(**params).get_params().copy()
        emulator.equations_ = PySREmulator.cache[get_key_for_cache_fit(X, y, params_emulator)][0]
        selected_complexity = emulator.selected_complexity
        try:
            selected_expr = emulator.selected_expr.copy()
        except TypeError:
            selected_expr = ""
        line_values.append((selected_complexity, selected_expr, params_emulator))
    selected_complexity_list, selected_expr_list, params_emulator_list = list(zip(*line_values))
    # Add columns to the DataFrame
    df_cv_results[SELECTED_COMPLEXITY_COLUMN_NAME] = selected_complexity_list
    df_cv_results[SELECTED_EXPR_COLUMN_NAME] = selected_expr_list
    df_cv_results[PARAMS_EMULATOR_COLUMN_NAME] = params_emulator_list
    return df_cv_results

