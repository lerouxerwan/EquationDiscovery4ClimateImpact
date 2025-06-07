import numpy as np

from emulator.emulator_validated_with_search import EmulatorValidatedWithSearch
from emulator.utils_hyperparameter_search.utils_column_names import PARAMS_EMULATOR_COLUMN_NAME, \
    get_cv_results_column_name, COMPLEXITY_COLUMN_NAME, FEATURE_INDEXES_COLUMN_NAME, \
    RMSE_VAL_COLUMN_NAME
from emulator.utils_hyperparameter_search.utils_df_results import get_selected_feature_indexes
from tests.emulator.utils_tests_emulator import run_three_main_functions_with_one_feature


def test_cv_results():
    # Run validation with the hyperparameter 'populations' that can have 2 values sampled between 10 and 20
    n_iter = 2
    emulator = EmulatorValidatedWithSearch(niterations=1, param_grid={'populations': [10, 20]}, n_iter=n_iter)
    run_three_main_functions_with_one_feature(emulator)
    # Check number of lines in df
    df = emulator.experiment_.df_cv_results
    assert len(df) == n_iter
    # Check the best selected complexity
    assert df[get_cv_results_column_name(emulator.model_selection, COMPLEXITY_COLUMN_NAME)].values[0] == 9
    # Check the selected feature indexes
    assert df[get_cv_results_column_name(emulator.model_selection, FEATURE_INDEXES_COLUMN_NAME)].values[0] == [0]
    # Check that it is well ranked
    validation_rmse_sorted_values = df[get_cv_results_column_name(emulator.model_selection, RMSE_VAL_COLUMN_NAME)].values
    for rmse1, rmse2 in zip(validation_rmse_sorted_values[:-1], validation_rmse_sorted_values[1:]):
        if not np.isnan(rmse2):
            assert rmse1 <= rmse2
    # Check that the best params are as expected
    best_params = df[PARAMS_EMULATOR_COLUMN_NAME].values[0]
    assert isinstance(best_params, dict)
    assert best_params['populations'] == 19
    assert best_params['niterations'] == 1
    # Remove folders at the end of the test
    emulator.experiment_.remove_folder()

def test_selected_feature_indexes():
    selected_variable_names = ['x0', 'x10', 'x84']
    # One test with variable_names = None
    assert get_selected_feature_indexes(selected_variable_names) == [0, 10, 84]
    # One test with specified variable_names
    variables_names = [f'x{2 * i}' for i in range(50)]
    assert get_selected_feature_indexes(selected_variable_names, variables_names) == [0, 5, 42]


