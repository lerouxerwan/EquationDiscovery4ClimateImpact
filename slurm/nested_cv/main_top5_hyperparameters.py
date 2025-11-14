
import sys
from collections import Counter

from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization_marginal.optimization_marginal import OptimizationMarginal
from optimization.optimization_random.optimization_random_zoo import OptimizationRandom_200, OptimizationRandom_5
from optimization.optmization_pipeline.optimization_pipeline_zoo_500 import OptimizationPipelineRandom, \
    OptimizationPipelineMarginalRandom
from optimization.utils_optimization import get_loss
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues, get_param_name_to_values
from plot.plot_diagnosis import plot_diagnosis
from plot.utils_metric.metric import metric_to_str
from utils.utils_log import log_info
from utils.utils_plot import show_or_save_plot


def main(show: bool = False):
    l = []
    opt_type = OptimizationMarginal
    nb_settings = 0
    for validation_size in [0.2, 0.25, 0.3]:
        for validation_split in [ValidationSplit.RANDOM, ValidationSplit.QUANTILE_WITH_BINNING, ValidationSplit.MIDDLE]:
            dataset = get_dataset(validation_size=validation_size, validation_split=validation_split)
            opt = opt_type('best', ParamNameToValues.DEFAULT_CENTRED_WO_OPERATORS, n_jobs=-1, timeout_in_seconds=60*60, interpretable_mode=True)
            _, param_name_to_values  = opt.run(dataset.X_train, dataset.y_train, dataset.validation_mask,
                     dataset.X_variable_names, dataset.X_units, dataset.y_units)
            top5_hyperparameters = list(param_name_to_values.keys())[:5]
            l.extend(top5_hyperparameters)
            nb_settings += 1
    # Analyze the frequency of top5 hyperparameters
    c = Counter(l)
    ax = plt.gca()
    param_names, occurrences = zip(*sorted(c.items(), key=lambda t: t[1], reverse=True))
    percentages = [100 * o / nb_settings for o in occurrences]
    ax.set_ylim(0, 100)
    x_values = range(len(c))
    ax.bar(x_values, percentages, width=0.5)
    ax.set_xticks(x_values)
    ax.set_xticklabels(param_names, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_ylabel(f'Selection rate in top5 hyperparameter for {nb_settings} optimization settings (%)')
    show_or_save_plot(f'main_top5_hyperparameters_{nb_settings}', show)


if __name__ == '__main__':
    main(show=True)