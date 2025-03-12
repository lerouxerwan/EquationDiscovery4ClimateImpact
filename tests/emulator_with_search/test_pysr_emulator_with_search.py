import numpy as np

from emulator_with_search.pysr_emulator_with_search import PySREmulatorWithSearch
from emulator_with_search.utils_param_grid.utils_scaling_factor import get_param_grid
from data.utils_search.utils_heredity_tree import add_heredity_link, \
    get_children, get_parent
from tests.data.utils_tests_dataset import load_X_and_y_ind_validation_for_test
from tests.emulator.utils_tests_emulator import load_climate_impact_emulator_with_search_for_test, \
    run_three_main_functions_with_one_feature


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
    search_cv = 'grid'
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

def test_search_experiment_tree():
    # Fit one emulator_parent and one emulator_child
    X, y, ind_validation = load_X_and_y_ind_validation_for_test()
    emulator_parent = PySREmulatorWithSearch(n_iter=1, scaling_factor=0, model_selection="custom", niterations=5)
    emulator_parent.fit(X, y, ind_validation=ind_validation)
    params_emulator_child = {**emulator_parent.search_experiment_.best_params, **{'adaptive_parsimony_scaling':500.}}
    emulator_child = PySREmulatorWithSearch(**params_emulator_child)
    emulator_child.fit(X, y, ind_validation=ind_validation)
    # Add heredity link (create parent and children files)
    add_heredity_link(emulator_child.search_experiment_, emulator_parent.search_experiment_)
    # Test get functions for parent
    assert get_parent(emulator_parent.search_experiment_) is None
    assert get_parent(emulator_child.search_experiment_).search_path == emulator_parent.search_experiment_.search_path
    # Test get functions for children
    assert len(get_children(emulator_child.search_experiment_)) == 0
    children_search_experiments = get_children(emulator_parent.search_experiment_)
    assert len(children_search_experiments) == 1
    assert children_search_experiments[0].search_path == emulator_child.search_experiment_.search_path
    # Remove folders
    for emulator in [emulator_parent, emulator_child]:
        emulator.search_experiment_.remove_folder()




