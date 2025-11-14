from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from emulator.emulator_with_search import EmulatorWithSearch
from emulator.utils_hyperparameter_search.utils_params_distribution import get_param_distributions
from optimization.optimization_marginal.optimization_marginal import OptimizationMarginal
from utils.utils_run import random_seed


def test_emulator_validation_with_search():
    dataset = get_dataset("NPP_season", 0.3, ValidationSplit.RCP_START)
    emulator = EmulatorWithSearch(niterations=5, n_iter=3, search_style='random',
                                  param_grid={'adaptive_parsimony_scaling': [520.0, 2080.0]},
                                  model_selection='validated')
    emulator.fit(dataset.X_train, dataset.y_train, validation_mask=dataset.validation_mask,
                 variable_names=dataset.X_variable_names, X_units=dataset.X_units, y_units=dataset.y_units)
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


def test_parenthesis_in_cv_results_for_interpretable_mode():
    dataset = get_dataset(validation_size=0.2, validation_split=ValidationSplit.RANDOM)
    param_name_to_values = {'niterations': [2, 4], 'ncycles_per_iteration': [2, 4]}
    opt = OptimizationMarginal('best', param_name_to_values, n_jobs=-1, timeout_in_seconds=60*60, interpretable_mode=True)
    top_emulator, _  = opt.run(dataset.X_train, dataset.y_train, dataset.validation_mask,
             dataset.X_variable_names, dataset.X_units, dataset.y_units)
    top_emulator.remove_folder()