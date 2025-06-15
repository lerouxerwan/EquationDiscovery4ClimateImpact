from itertools import combinations
from random import randint

from sklearn.model_selection import ParameterSampler

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from emulator.emulator import Emulator
from emulator.utils_hyperparameter_search.utils_params_distribution import get_param_distributions
from plot.workflow import fit
from projects.simple_paper.search_strategy import get_params_search, SearchStrategy
from utils.utils_log import log_info
from utils.utils_run import random_seed




param_names_with_float_values = {'adaptive_parsimony_scaling',
'fraction_replaced', 'fraction_replaced_hof',
'weight_add_node', 'weight_insert_node', 'weight_delete_node',
'weight_do_nothing', 'weight_mutate_constant', 'weight_mutate_operator',
'weight_swap_operands', 'weight_rotate_tree', 'weight_randomize',
'weight_simplify', 'crossover_probability',
"perturbation_factor", "probability_negate_constant"}

def get_param_bug():
    param_grid = get_params_search(SearchStrategy.ALL_EXCLUDING_3)['param_grid']
    param_distributions = get_param_distributions(param_grid)
    parameter_sampler = ParameterSampler(param_distributions, 10, random_state=random_seed)
    for i, params in enumerate(parameter_sampler):
        if i==1:
            return params

def main_bug_infinite_search(fast: bool = False):
    params_bug = get_param_bug()
    dataset = Dataset("NPP_season.csv", "RCP85", "RCP45", 0.3, ValidationSplit.START)
    # Compute of couple combination to try
    param_names_for_couple_search = [param_name for param_name in params_bug.keys()
                                     if param_name not in param_names_with_float_values]
    couples = list(combinations(param_names_for_couple_search, 2))
    name_to_duration = {}
    for j, couple in enumerate(couples, 1):
        log_info(f'')
        param_name1, param_name2 = couple
        log_info(f'Couple #{j}/{len(couples)}, Run with {param_name1} and {param_name2}')
        niterations = 2 if fast else 100
        couples_values = {param_name: params_bug[param_name] for param_name in [param_name1, param_name2]}
        log_info(str(couples_values))
        emulator = Emulator(niterations=niterations, **couples_values)
        fit(emulator, dataset)
        print(emulator.experiment_.duration)
        # name_to_duration[str(couples_values)] = emulator.dura
        # print(param_name, param_value, param_grid[param_name])

if __name__ == '__main__':
    main_bug_infinite_search(fast=False)

