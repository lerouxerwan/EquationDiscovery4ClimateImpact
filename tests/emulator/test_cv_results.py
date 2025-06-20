import numpy as np

from emulator.emulator_with_search import EmulatorWithSearch
from emulator.utils_hyperparameter_search.utils_column_names import RMSE_VALIDATION_COLUMN_NAME
from tests.data.utils_tests_dataset import load_X_and_y_and_validation_mask_for_test


def test_cv_results():
    # Run validation with the hyperparameter 'populations' that can have 2 values sampled between 10 and 20
    n_iter = 2
    emulator = EmulatorWithSearch(niterations=1, param_grid={'populations': [10, 20]}, n_iter=n_iter)
    X, y, validation_mask = load_X_and_y_and_validation_mask_for_test()
    emulator.fit(X, y, validation_mask)
    # Check number of lines in df
    df = emulator.experiment_.df_cv_results
    assert len(df) == n_iter
    # Check the top selected complexity
    assert emulator.selected_complexity == 9
    # Check that it is well ranked
    validation_rmse_sorted_values = df[RMSE_VALIDATION_COLUMN_NAME].values
    for rmse1, rmse2 in zip(validation_rmse_sorted_values[:-1], validation_rmse_sorted_values[1:]):
        if not np.isnan(rmse2):
            assert rmse1 <= rmse2
    # Check that the emulator params are as expected
    assert emulator.populations == 19
    assert emulator.niterations == 1
    # Remove folders at the end of the test
    emulator.experiment_.remove_folder()



