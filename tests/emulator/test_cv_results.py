import numpy as np

from emulator.emulator_with_search import EmulatorWithSearch
from tests.data.utils_tests_dataset import load_X_and_y_and_validation_mask_for_test


def test_cv_results():
    # Run validation with the hyperparameter 'populations' that can have 2 values sampled between 10 and 20
    n_iter = 2
    emulator = EmulatorWithSearch(niterations=1, param_grid={'populations': [10, 20]}, n_iter=n_iter)
    X, y, validation_mask = load_X_and_y_and_validation_mask_for_test()
    emulator.fit(X, y, validation_mask)
    # Check number of lines in df
    experiment = emulator.experiment_
    df = experiment.df_cv_results
    assert len(df) == n_iter
    # Check the top selected complexity
    assert experiment.top_complexity == 9
    # Check that it is well ranked
    validation_rmse_sorted_values = df[experiment.rmse_validation_column_name].values
    for rmse1, rmse2 in zip(validation_rmse_sorted_values[:-1], validation_rmse_sorted_values[1:]):
        if not np.isnan(rmse2):
            assert rmse1 <= rmse2
    # Check that the top params are as expected
    top_params = experiment.top_params
    assert isinstance(top_params, dict)
    assert top_params['populations'] == 19
    assert top_params['niterations'] == 1
    # Remove folders at the end of the test
    experiment.remove_folder()



