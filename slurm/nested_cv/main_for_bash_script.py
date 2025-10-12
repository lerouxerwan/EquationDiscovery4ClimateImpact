import sys

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization_marginal.optimization_marginal import OptimizationMarginal
from optimization.utils_nested_cv.run_nested_cv import run_nested_cv
from optimization.utils_optimization import get_loss
from plot.utils_metric.metric import Metric


def main():
    if len(sys.argv) > 1:
        indices = [int(sys.argv[i]) for i in range(1, 3)]
    else:
        indices = [0, 4]
    print(f'Run with indices={indices}')

    dataset = Dataset("NPP_season.csv", "RCP85", "RCP45", 0.2, ValidationSplit.QUANTILE_WITH_BINNING)
    model_selection_list = ['best', 'validated']
    model_selection = model_selection_list[indices[0]]

    param_name_to_values = {'niterations': [2, 4], 'populations': [2, 4], 'ncycles_per_iteration': [2, 4]}
    opt = OptimizationMarginal('best', param_name_to_values)
    opt.run(dataset.X_train, dataset.y_train, dataset.validation_mask,
             dataset.X_variable_names, dataset.X_units, dataset.y_units)
    loss = get_loss(opt, dataset.X_train, dataset.y_train, dataset.validation_mask,
             dataset.X_variable_names, dataset.X_units, dataset.y_units,
             Metric.RMSE, dataset.X_test, dataset.y_test)
    print(loss)

    # Optimization with double search
    # nb_top_hyperparameters = indices[1]
    # opt = OptimizationMarginalThenRandom(model_selection, ParamsValues.DEFAULT_CENTRED, nb_top_hyperparameters)

    # Optimization with optuna
    # nb_trials = indices[1]
    # opt= OptimizationOptuna(model_selection, ParamsValues.DEFAULT_CENTRED, nb_trials)

    # run_nested_cv(dataset, opt)


if __name__ == '__main__':
    main()