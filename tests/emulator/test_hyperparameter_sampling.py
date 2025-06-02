from emulator.emulator_validated_with_search import EmulatorValidatedWithSearch
from emulator.utils_hyperparameter_search.utils_search_cv import get_random_params_list_from_param_grid
from emulator.utils_hyperparameter_search.utils_scaling_factor import get_param_grid


def test_scaling_factor_for_grid_search():
    scaling_factor = 2
    search_cv = 'grid'
    n_iter = 5
    # get_grid_search without specifying 'param_list_to_optimize'
    emulator = EmulatorValidatedWithSearch(niterations=10)
    param_grid = get_param_grid(emulator, scaling_factor, search_cv, n_iter)
    assert len(param_grid) == 1
    param_values = param_grid['niterations']
    assert (param_values[0] == 5) and (param_values[2] == 10) and (param_values[-1] == 20)
    # get_grid_search specifying 'param_list_to_optimize'
    emulator = EmulatorValidatedWithSearch(adaptive_parsimony_scaling=1000., fraction_replaced_hof=0.01)
    param_list_to_optimize = ['adaptive_parsimony_scaling', 'fraction_replaced_hof']
    param_grid = get_param_grid(emulator, scaling_factor, search_cv, n_iter, param_list_to_optimize)
    assert len(param_grid) == 2
    param_values = param_grid['adaptive_parsimony_scaling']
    assert (param_values[0] == 500.) and (param_values[2] == 1000.) and (param_values[-1] == 2000.)
    param_values = param_grid['fraction_replaced_hof']
    assert (param_values[0] == 0.005) and (param_values[2] == 0.01) and (param_values[-1] == 0.02)

def test_random_state_for_hyperparameter_search():
    emulator = EmulatorValidatedWithSearch(niterations=10)
    for _ in range(2):
        random_params_list = get_random_params_list_from_param_grid(emulator.param_grid, emulator.n_iter)
        sum_niterations = sum([params['niterations'] for params in random_params_list])
        assert sum_niterations == 227


