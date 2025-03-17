import numpy as np

from emulator_with_search.pysr_emulator_with_search import PySREmulatorWithSearch
from emulator_with_search.utils_cv_results.utils_df_results import RMSE_VALIDATION_COLUMN_NAME, \
    PARAMS_EMULATOR_COLUMN_NAME, METRIC_COLUMN_NAME, SELECTED_COMPLEXITY_COLUMN_NAME
from tests.emulator.utils_tests_emulator import run_three_main_functions_with_one_feature


def test_df_cv_results():
    # Run validation with the hyperparameter 'populations' that can have 2 values sampled between 10 and 20
    n_iter = 2
    emulator = PySREmulatorWithSearch(niterations=1, param_grid={'populations': [10, 20]}, n_iter=n_iter)
    run_three_main_functions_with_one_feature(emulator)
    # Check number of lines in df
    df = emulator.search_experiment_.df_cv_results
    assert len(df) == n_iter
    # Check the best selected complexity
    print(df[SELECTED_COMPLEXITY_COLUMN_NAME].values)
    assert df[SELECTED_COMPLEXITY_COLUMN_NAME].values[0] == 9
    # Check that it is well ranked
    validation_rmse_sorted_values = df[RMSE_VALIDATION_COLUMN_NAME].values
    for rmse1, rmse2 in zip(validation_rmse_sorted_values[:-1], validation_rmse_sorted_values[1:]):
        if not np.isnan(rmse2):
            assert rmse1 <= rmse2
    # Check that the best params are as expected
    best_params = df[PARAMS_EMULATOR_COLUMN_NAME].values[0]
    assert isinstance(best_params, dict)
    assert best_params['populations'] == 19
    assert best_params['niterations'] == 1
    # Remove folders at the end of the test
    emulator.search_experiment_.remove_folder()
