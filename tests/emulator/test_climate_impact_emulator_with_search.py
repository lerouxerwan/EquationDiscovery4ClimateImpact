import numpy as np

from tests.emulator.utils_tests_emulator import load_climate_impact_emulator_with_search_for_test, run_three_main_functions


def test_one_hyperparameter():
    # Run validation with the hyperparameter 'populations' that can have 2 values sampled between 10 and 20
    emulator = load_climate_impact_emulator_with_search_for_test(param_grid={'populations': [10, 20]},
                                                                 n_iter=2, save_or_load_csv_of_search_results=False)
    run_three_main_functions(emulator)
    total_loss_expected = -32.906312030116666
    total_loss_computed = float(emulator.df_cv_results_ranked_['mean_train_MSE'].values.sum())
    np.testing.assert_almost_equal(total_loss_computed, total_loss_expected)
