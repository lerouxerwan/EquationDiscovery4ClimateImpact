import sys

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization_double_search import OptimizationDoubleSearch
from optimization.utils_nested_cv import run_nested_cv
from optimization.utils_params.utils_params_values import ParamsValues


def main():
    if len(sys.argv) > 1:
        indices = [int(sys.argv[i]) for i in range(1, 3)]
        fast = False
    else:
        indices = [0, 4]
        fast = True
    print(f'Run with indices={indices}')

    dataset = Dataset("NPP_season.csv", "RCP85", "RCP45", 0.2, ValidationSplit.QUANTILE_WITH_BINNING)
    model_selection_list = ['best', 'validated']
    model_selection = model_selection_list[indices[0]]

    # Optimization with double search
    nb_top_hyperparameters = indices[1]
    opt = OptimizationDoubleSearch(model_selection, ParamsValues.DEFAULT_CENTRED, nb_top_hyperparameters)

    # Optimization with optuna
    # nb_trials = indices[1]
    # opt= OptimizationOptuna(model_selection, ParamsValues.DEFAULT_CENTRED, nb_trials)

    run_nested_cv(dataset, opt, fast)


if __name__ == '__main__':
    main()