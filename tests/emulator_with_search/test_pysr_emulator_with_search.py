import numpy as np
from sklearn.model_selection import GridSearchCV

from emulator_with_search.pysr_emulator_with_search import PySREmulatorWithSearch
from emulator_with_search.utils_search.utils_scaling_factor import get_param_grid
from tests.utils_tests_emulator import load_climate_impact_emulator_with_search_for_test, run_three_main_functions_with_one_feature


def test_one_hyperparameter():
    # Run validation with the hyperparameter 'populations' that can have 2 values sampled between 10 and 20
    emulator = load_climate_impact_emulator_with_search_for_test(param_grid={'populations': [10, 20]}, n_iter=2)
    run_three_main_functions_with_one_feature(emulator)
    total_loss_expected = -63.017190749671634
    total_loss_computed = float(emulator.search_experiment_.df_cv_results_ranked_augmented['mean_test_MSE'].sum())
    emulator.search_experiment_.remove_folder()
    np.testing.assert_almost_equal(total_loss_computed, total_loss_expected)


def test_scaling_factor_for_grid_search():
    scaling_factor = 2
    search_cv = GridSearchCV
    n_iter = 5
    # get_grid_search without specifying 'param_list_to_optimize'
    emulator = PySREmulatorWithSearch(niterations=10)
    param_grid = get_param_grid(emulator, scaling_factor, search_cv, n_iter)
    assert len(param_grid) == 1
    fraction_replaced_hof_for_optimization = param_grid['niterations']
    assert fraction_replaced_hof_for_optimization[0] == 5
    assert fraction_replaced_hof_for_optimization[2] == 10
    assert fraction_replaced_hof_for_optimization[-1] == 20
    # get_grid_search specifying 'param_list_to_optimize'
    emulator = PySREmulatorWithSearch(adaptive_parsimony_scaling=1000., fraction_replaced_hof=0.01)
    param_list_to_optimize = ['adaptive_parsimony_scaling', 'fraction_replaced_hof']
    param_grid = get_param_grid(emulator, scaling_factor, search_cv, n_iter, param_list_to_optimize)
    assert len(param_grid) == 2
    fraction_replaced_hof_for_optimization = param_grid['adaptive_parsimony_scaling']
    assert fraction_replaced_hof_for_optimization[0] == 500.
    assert fraction_replaced_hof_for_optimization[2] == 1000.
    assert fraction_replaced_hof_for_optimization[-1] == 2000.
    fraction_replaced_hof_for_optimization = param_grid['fraction_replaced_hof']
    assert fraction_replaced_hof_for_optimization[0] == 0.005
    assert fraction_replaced_hof_for_optimization[2] == 0.01
    assert fraction_replaced_hof_for_optimization[-1] == 0.02

