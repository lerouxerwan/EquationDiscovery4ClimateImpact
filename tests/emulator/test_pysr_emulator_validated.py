import numpy as np

from tests.emulator.utils_tests_emulator import load_pysr_emulator_validated_for_test, run_three_main_functions


def test_pysr_emulator_validated_for_one_hyperparameter_and_multiple_jobs():
    # Run validation with the hyperparameter 'populations' that can have 2 values sampled between 10 and 20
    emulator = load_pysr_emulator_validated_for_test(param_grid={'populations': [10, 20]}, n_iter=2, n_jobs=2)
    run_three_main_functions(emulator)
    total_loss_with_1_job = -1.2948700141031671e-09
    total_loss_with_2_jobs = float(emulator.df_ranked_results_['mean_train_MSE'].values.sum())
    np.testing.assert_almost_equal(total_loss_with_2_jobs, total_loss_with_1_job)
