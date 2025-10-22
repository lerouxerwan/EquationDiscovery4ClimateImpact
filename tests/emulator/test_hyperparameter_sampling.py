from emulator.emulator_with_search import EmulatorWithSearch
from emulator.utils_hyperparameter_search.utils_search_cv import get_random_params_list_from_param_grid



def test_random_state_for_hyperparameter_search():
    emulator = EmulatorWithSearch(param_grid={'niterations': [10, 20]})
    for _ in range(2):
        random_params_list = get_random_params_list_from_param_grid(emulator.param_grid, emulator.n_iter)
        sum_niterations = sum([params['niterations'] for params in random_params_list])
        assert sum_niterations == 147


