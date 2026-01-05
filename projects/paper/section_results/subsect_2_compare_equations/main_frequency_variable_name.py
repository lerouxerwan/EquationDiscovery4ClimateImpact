from typing import OrderedDict, Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from emulator.utils_variable_names import get_variable_signed_names
from optimization.optimization_marginal.optimization_marginal import OptimizationMarginal
from optimization.optimization_random.optimization_random_zoo import OptimizationRandom_200, OptimizationRandom_4, \
    OptimizationRandom_500
from optimization.optmization_pipeline.optimization_pipeline_zoo_500 import OptimizationPipelineMarginalRandom, \
    OptimizationPipelineRandom
from optimization.utils_optimization import get_loss
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from projects.paper.utils_paper import validation_sizes, opt_type, validation_splits
from utils.utils_latex import print_df_latex
from utils.utils_log import log_info
from utils.utils_plot import show_or_save_plot


def main_plot_frequency_variable(show: bool = False,
                                 signed_name: bool = True):
    title = f'Compare split for optimization {opt_type.__name__} with signed_name = {signed_name}'
    counter_variable_names = Counter()
    nb_loop = 0

    # First graph
    ax = plt.gca()
    validation_split_to_variable_names_for_selected_equation = OrderedDict()
    validation_split_to_min_RMSE = {validation_split: np.inf for validation_split in validation_splits}
    for validation_size, color in zip(validation_sizes, ['yellow', 'orange', 'red']):
        opt = opt_type('best', ParamNameToValues.DEFAULT_CENTRED_WO_OPERATORS, n_jobs=-1, timeout_in_seconds=60 * 60, interpretable_mode=True)
        validation_split_to_rmse_test_for_selected_equation = OrderedDict()
        for validation_split in validation_splits:
                dataset = get_dataset(validation_size=validation_size, validation_split=validation_split)
                emulator = opt.get_top_emulator(dataset.X_train, dataset.y_train, dataset.validation_mask,
                                     dataset.X_variable_names, dataset.X_units, dataset.y_units)
                variable_names = emulator.selected_variable_signed_names if signed_name else emulator.selected_variable_names
                rmse_test = get_loss(opt, dataset.X_train, dataset.y_train, dataset.validation_mask,
                                     dataset.X_variable_names, dataset.X_units, dataset.y_units,
                                     dataset.X_test, dataset.y_test)
                log_info(f'RMSE test = {rmse_test} for {validation_split}')
                validation_split_to_rmse_test_for_selected_equation[validation_split] = rmse_test
                if rmse_test < validation_split_to_min_RMSE[validation_split]:
                    validation_split_to_variable_names_for_selected_equation[validation_split] =  '\n'.join(variable_names)

                # Information for the second graph
                counter_variable_names.update(variable_names)
                nb_loop += 1

        ax.plot(validation_split_to_rmse_test_for_selected_equation.keys(), validation_split_to_rmse_test_for_selected_equation.values(),
                label=f'validation_size={int(100 * validation_size)}%', color=color)

    ax.set_xlabel('Validation split')
    ax.set_ylabel(f'RMSE test ({dataset.y_units[0]})')
    ax_twin = ax.twiny()
    ax_twin.set_xticks(ax.get_xticks())
    ax_twin.set_xlim(ax.get_xlim())
    ax_twin.set_xticklabels(validation_split_to_variable_names_for_selected_equation.values())
    ax.set_ylim((1.4, 2. ))
    ax.grid()
    # ax.set_title(title)
    ax.legend()
    ax.tick_params(axis='x', which='major', labelsize=8)
    ax_twin.tick_params(axis='x', which='major', labelsize=5)
    ax_twin.set_xlabel('Variable names in the equation that minimizes RMSE test')
    show_or_save_plot('_'.join(title.split()), show)

    # Second graph
    ax = plt.gca()
    sorted_items = sorted(counter_variable_names.items(), key=lambda x: x[1], reverse=True)
    labels, y_values = zip(*sorted_items)
    x_values = np.arange(len(labels))
    y_values = 100 * np.array(y_values) / nb_loop
    print(y_values)
    ax.bar(x_values, y_values, color='skyblue')
    ax.set_xlabel('Variable names')
    ax.set_ylabel('Frequency (%)')
    ax.set_xticks(x_values)
    labels = [label.replace('AnnSea', 'Annual') for label in labels]
    labels = ['$' + label.replace('_', '_{') + '}$' for label in labels]
    ax.set_xticklabels(labels, rotation=45, ha='right', rotation_mode='anchor')
    # ax.tick_params(axis='x', which='major', labelsize=8, labelrotation=45)
    ax.grid(axis='y')
    plot_name = 'frequency_variable_signed_names' if signed_name else 'frequency_variable_names'
    plot_name += f' for {opt_type.__name__}'
    show_or_save_plot(plot_name, show)



if __name__ == '__main__':
    main_plot_frequency_variable()


