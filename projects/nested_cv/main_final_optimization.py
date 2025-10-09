from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization import Optimization
from optimization.optimization_baseline import OptimizationBaseline
from optimization.optimization_marginal_grid import OptimizationMarginalGrid
from optimization.optimization_marginal_grid_then_random import OptimizationMarginalGridThenRandom
from optimization.utils_nested_cv.compare_nested_cv import compare_nested_cv
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues, param_names
from plot.plot_diagnosis import plot_diagnosis

dataset= Dataset("NPP_season.csv", "RCP85", "RCP45", 0.2, ValidationSplit.QUANTILE_WITH_BINNING)

def main_plot_diagnosis_top_emulator():
    for param_name in param_names[:]:
        opt = OptimizationMarginalGrid('best', ParamNameToValues.DEFAULT_CENTRED, param_name)
        emulator = opt.get_top_emulator(dataset.X_train, dataset.y_train, dataset.validation_mask,
                                        dataset.X_variable_names, dataset.X_units, dataset.y_units)

    # for nb_top in range(1, 10):
    #     opt = OptimizationMarginalGridThenRandom('best', ParamsValues.DEFAULT_CENTRED, nb_top)
        # opt = OptimizationBaseline('best')
        # opt = OptimizationMarginalGrid('best', ParamsValues.DEFAULT_CENTRED, 'weight_swap_operands')
        # emulator = opt.get_top_emulator(dataset.X_train, dataset.y_train, dataset.validation_mask,
        #                                 dataset.X_variable_names, dataset.X_units, dataset.y_units)
        # plot_diagnosis(emulator, dataset)

def main_compare_nested_cv():
    opt_list: list[Optimization] = []
    for model_selection in ['best']:
        opt_list.append(OptimizationBaseline(model_selection))
        # Add optimization marginal
        # for param_name in param_names[:]:
        # for param_name in ['weight_swap_operands']:
        for param_name in ['weight_optimize', 'population_size']:
            opt_list.append(OptimizationMarginalGrid(model_selection, ParamNameToValues.DEFAULT_CENTRED, param_name))
        #  Add optimization marginal then random
        for nb_top_hyperparameters in [6, 10]:
            opt_list.append(OptimizationMarginalGridThenRandom(model_selection, ParamNameToValues.DEFAULT_CENTRED, nb_top_hyperparameters))
        # Add optimization couple
        # for param_name_1 in ['weight_swap_operands']:
        #     for param_name_2 in ['niterations']:
        #             opt_list.append(OptimizationCoupleRandom(model_selection, ParamsValues.DEFAULT_CENTRED, param_name_1, param_name_2))
        compare_nested_cv(dataset, opt_list)


if __name__ == '__main__':
    # main_compare_nested_cv()
    main_plot_diagnosis_top_emulator()
