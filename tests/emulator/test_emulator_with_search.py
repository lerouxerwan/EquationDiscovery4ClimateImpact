from data.utils_dataset.npp_season_v1 import dataset_npp_season_v1
from emulator.emulator_with_search import EmulatorWithSearch
from emulator.utils_hyperparameter_search.utils_params_distribution import get_param_distributions
from utils.utils_run import random_seed


def test_emulator_validation_with_search():
    dataset = dataset_npp_season_v1
    emulator = EmulatorWithSearch(niterations=5, n_iter=3, search_style='random', scaling_factor=2,
                                  model_selection='validated',
                                  param_list_to_optimize=['adaptive_parsimony_scaling'])
    emulator.fit(dataset.X_train, dataset.y_train, validation_mask=dataset.validation_mask,
                 variable_names=dataset.X_variables_names, X_units=dataset.X_units, y_units=dataset.y_units)
    assert emulator.selected_complexity == 27
    emulator.remove_folder()

def test_random_sampling():
    param_name = 'niterations'
    param_grid = {param_name: [50, 200]}
    param_distribution = get_param_distributions(param_grid)[param_name]
    ten_samples = param_distribution.rvs(10, random_seed)
    twenty_samples = param_distribution.rvs(100, random_seed)
    # Check that the first 10 samples of "twenty_samples" are the same that if we were sampling ony 10 samples
    for sample1, sample2 in zip(ten_samples, twenty_samples):
        assert sample1 == sample2
