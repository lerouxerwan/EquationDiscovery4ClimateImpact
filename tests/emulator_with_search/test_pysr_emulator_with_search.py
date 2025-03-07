import numpy as np

from tests.utils_tests_emulator import load_climate_impact_emulator_with_search_for_test, run_three_main_functions_with_one_feature


def test_one_hyperparameter():
    # Run validation with the hyperparameter 'populations' that can have 2 values sampled between 10 and 20
    emulator = load_climate_impact_emulator_with_search_for_test(param_grid={'populations': [10, 20]}, n_iter=2)
    run_three_main_functions_with_one_feature(emulator)
    total_loss_expected = -63.017190749671634
    total_loss_computed = float(emulator.search_experiment_.df_cv_results_ranked_augmented['mean_test_MSE'].sum())
    emulator.search_experiment_.remove_folder()
    np.testing.assert_almost_equal(total_loss_computed, total_loss_expected)
