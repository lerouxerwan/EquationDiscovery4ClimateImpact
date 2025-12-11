from collections import OrderedDict
from itertools import product

from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit, get_validation_label
from optimization.optimization_marginal.optimization_marginal import OptimizationMarginal
from optimization.optmization_pipeline.optimization_pipeline_zoo_500 import OptimizationPipelineRandom, \
    OptimizationPipelineMarginalRandom
from optimization.utils_optimization import get_loss
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from plot.by_split.plot_loss_vs_complexity import load_bar_attributes
from plot.by_split.utils_axis import set_log_y_axis
from plot.utils_metric.metric import Metric
from utils.utils_log import log_info
from utils.utils_plot import show_or_save_plot
from itertools import product

from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization_marginal.optimization_marginal import OptimizationMarginal
from optimization.optmization_pipeline.optimization_pipeline_zoo_500 import OptimizationPipelineRandom, \
    OptimizationPipelineMarginalRandom
from optimization.utils_optimization import get_loss
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from plot.by_split.plot_loss_vs_complexity import load_bar_attributes
from utils.utils_log import log_info
from utils.utils_plot import show_or_save_plot


def main_check_number_of_crashes(fast: bool, show: bool):
    validation_sizes = [0.2, 0.25, 0.3]
    validation_splits = [ValidationSplit.RANDOM, ValidationSplit.QUANTILE_WITH_BINNING, ValidationSplit.MIDDLE]
    if fast:
        validation_sizes, validation_splits = validation_sizes[:2], validation_splits[:1]
    datasets = [get_dataset(validation_size=validation_size, validation_split=validation_split)
        for validation_split, validation_size in product(validation_splits, validation_sizes)]
    opt_types = [OptimizationPipelineRandom][:]
    opt_name_to_list_percentage_of_crash = OrderedDict()
    for opt_type in opt_types:
        opt = opt_type('best', ParamNameToValues.DEFAULT_CENTRED_WO_OPERATORS, n_jobs=1, timeout_in_seconds=60 * 60, interpretable_mode=True)
        list_percentage_of_crash = []
        for dataset in datasets:
            top_emulator, _ = opt.run(dataset.X_train, dataset.y_train, dataset.validation_mask,
                                      dataset.X_variable_names, dataset.X_units, dataset.y_units)
            ind_crash = top_emulator.run_.ind_crashes
            percentage_of_crash = float(100 * ind_crash.sum() / len(ind_crash))
            list_percentage_of_crash.append(percentage_of_crash)
        opt_name_to_list_percentage_of_crash[type(opt).__name__] = list_percentage_of_crash
    # Print everything
    for key, values in opt_name_to_list_percentage_of_crash.items():
        print(f'Percentages of crashes for {key} = {values}')


if __name__ == '__main__':
    main_check_number_of_crashes(False, True)