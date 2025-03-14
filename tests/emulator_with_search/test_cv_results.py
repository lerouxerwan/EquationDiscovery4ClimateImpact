import numpy as np

from emulator.utils_attributes.utils_threshold import NB_THRESHOLDS
from emulator_with_search.pysr_emulator_with_search import PySREmulatorWithSearch
from tests.emulator.utils_tests_emulator import run_three_main_functions_with_one_feature


def test_df_cv_results_ranked_and_augmented():
    # Run validation with the hyperparameter 'populations' that can have 2 values sampled between 10 and 20
    n_iter = 2
    emulator = PySREmulatorWithSearch(niterations=1, param_grid={'populations': [10, 20]}, n_iter=n_iter)
    run_three_main_functions_with_one_feature(emulator)
    # Check number of lines in df
    df = emulator.search_experiment_.df_cv_results_ranked_and_augmented
    assert len(df) == n_iter * NB_THRESHOLDS
    # Check the total loss
    np.testing.assert_almost_equal(float(df['mean_test_MSE'].sum()), -63.017190749671634)
    # Check the best selected complexity
    assert df['selected_complexity'].values[0] == 9
    # Check that it is well ranked
    validation_rmse_sorted_values = df['RMSE_validation'].values
    for rmse1, rmse2 in zip(validation_rmse_sorted_values[:-1], validation_rmse_sorted_values[1:]):
        if not np.isnan(rmse2):
            assert rmse1 <= rmse2
    # Remove folders at the end of the test
    emulator.search_experiment_.remove_folder()
