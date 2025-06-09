import numpy as np

from emulator.emulator_validated_with_search import EmulatorValidatedWithSearch
from emulator.utils_hyperparameter_search.utils_column_names import PARAMS_EMULATOR_COLUMN_NAME
from tests.emulator.utils_tests_emulator import run_three_main_functions_with_one_feature


def test_cv_results():
    # Run validation with the hyperparameter 'populations' that can have 2 values sampled between 10 and 20
    n_iter = 2
    emulator = EmulatorValidatedWithSearch(niterations=1, param_grid={'populations': [10, 20]}, n_iter=n_iter)
    run_three_main_functions_with_one_feature(emulator)
    # Check number of lines in df
    experiment = emulator.experiment_
    df = experiment.df_cv_results
    assert len(df) == n_iter
    # Check the top selected complexity
    assert experiment.top_complexity == 9
    # Check that it is well ranked
    validation_rmse_sorted_values = df[experiment.rmse_val_column_name].values
    for rmse1, rmse2 in zip(validation_rmse_sorted_values[:-1], validation_rmse_sorted_values[1:]):
        if not np.isnan(rmse2):
            assert rmse1 <= rmse2
    # Check that the best params are as expected
    best_params = df[PARAMS_EMULATOR_COLUMN_NAME].values[0]
    assert isinstance(best_params, dict)
    assert best_params['populations'] == 19
    assert best_params['niterations'] == 1
    # Remove folders at the end of the test
    experiment.remove_folder()



