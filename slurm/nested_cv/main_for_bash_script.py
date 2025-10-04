import sys

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization_couple_random import OptimizationCoupleRandom
from optimization.optimization_marginal_grid_then_random import OptimizationMarginalGridThenRandom
from optimization.utils_nested_cv.run_nested_cv import run_nested_cv
from optimization.utils_params.utils_params_values import ParamsValues, param_names


def main():
    if len(sys.argv) > 1:
        indices = [int(sys.argv[i]) for i in range(1, 3)]
    else:
        indices = [0, 4]
    print(f'Run with indices={indices}')

    dataset = Dataset("NPP_season.csv", "RCP85", "RCP45", 0.2, ValidationSplit.QUANTILE_WITH_BINNING)
    model_selection_list = ['best', 'validated']
    model_selection = model_selection_list[indices[0]]

    # Optimization with double search
    # nb_top_hyperparameters = indices[1]
    # opt = OptimizationMarginalThenRandom(model_selection, ParamsValues.DEFAULT_CENTRED, nb_top_hyperparameters)

    # Optimization with couple search
    # param_name_1 = 'weight_swap_operands'
    # param_names_without_param_name_1 = param_names[:]
    # param_names_without_param_name_1.remove(param_name_1)
    # param_name_2 = param_names_without_param_name_1[indices[2]]
    # opt = OptimizationCoupleRandom(model_selection, ParamsValues.DEFAULT_CENTRED, param_name_1, param_name_2)

    # Optimization with optuna
    # nb_trials = indices[1]
    # opt= OptimizationOptuna(model_selection, ParamsValues.DEFAULT_CENTRED, nb_trials)

    run_nested_cv(dataset, opt)


if __name__ == '__main__':
    main()