from typing import OrderedDict, Counter

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from emulator.ordered_equation import OrderedEquation
from emulator.utils_variable_names import get_variable_signed_names
from optimization.optimization_marginal.optimization_marginal import OptimizationMarginal
from optimization.optimization_random.optimization_random_zoo import OptimizationRandom_200, OptimizationRandom_4, \
    OptimizationRandom_500
from optimization.optmization_pipeline.optimization_pipeline_zoo_500 import OptimizationPipelineMarginalRandom, \
    OptimizationPipelineRandom
from optimization.utils_optimization import get_loss
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from projects.paper.utils_paper import validation_sizes, opt_type, validation_splits, opt
from utils.utils_latex import print_df_latex
from utils.utils_log import log_info
from utils.utils_plot import show_or_save_plot
import matplotlib.patches as mpatches

def main_plot_signed_frequency(show: bool = False):
    # Build the dictionary with the weights
    variable_signed_name_to_weights = dict()
    nb_loop = len(validation_sizes) * len(validation_splits)
    for validation_size in validation_sizes:
        for validation_split in validation_splits:
                dataset = get_dataset(validation_size=validation_size, validation_split=validation_split)
                emulator = opt.get_top_emulator(dataset.X_train, dataset.y_train, dataset.validation_mask,
                                     dataset.X_variable_names, dataset.X_units, dataset.y_units)
                ordered_equation = OrderedEquation(emulator.selected_expressions[0], dataset.X_train, dataset.X_variable_names)
                # Augment the dictionary with the weights
                for variable_signed_name, weight in ordered_equation.variable_signed_name_to_weight.items():
                    if variable_signed_name in variable_signed_name_to_weights:
                        variable_signed_name_to_weights[variable_signed_name].append(weight)
                    else:
                        variable_signed_name_to_weights[variable_signed_name] = [weight]
    # Build additional dictionary
    variable_signed_name_to_number = {variable_signed_name: len(weights) for variable_signed_name, weights in variable_signed_name_to_weights.items()}
    sorted_variable_signed_name =  [c[0] for c in sorted(variable_signed_name_to_number.items(), key=lambda x: x[1], reverse=True)]
    variable_signed_name_to_percentage = {variable_signed_name: 100 * number / nb_loop for variable_signed_name, number in variable_signed_name_to_number.items()}
    variable_signed_name_to_signed_percentage = {variable_signed_name: (+1 if variable_signed_name[0] == '+' else -1) * percentage
                                                                        for variable_signed_name, percentage in variable_signed_name_to_percentage.items()}

    # Plot
    fig, ax = plt.subplots()
    labels = sorted_variable_signed_name
    x_values = np.arange(len(labels))
    y_values = [variable_signed_name_to_signed_percentage[variable_signed_name] for variable_signed_name in sorted_variable_signed_name]
    sorted_average_weight = [float(np.mean(variable_signed_name_to_weights[variable_signed_name])) for variable_signed_name in sorted_variable_signed_name]
    assert all([0 <= weight <= 1 for weight in sorted_average_weight])
    cmap_original = plt.get_cmap('Greens')
    cmap_new = LinearSegmentedColormap.from_list(
        'Greens_up',
        cmap_original(np.linspace(0.2, 1, 256))
    )
    colors = cmap_new(sorted_average_weight)
    sm = ScalarMappable(cmap=cmap_new, norm=plt.Normalize(vmin=0, vmax=1))
    bars = ax.bar(x_values, y_values, color=colors)
    ax.set_xlabel('Variable names')
    ax.set_ylabel('Frequency (%)')
    ax.set_xticks(x_values)
    labels = [label.replace('AnnSea', 'Annual') for label in labels]
    labels = ['$' + label.replace('_', '_{') + '}$' for label in labels]
    ax.set_xticklabels(labels, rotation=45, ha='right', rotation_mode='anchor')
    plot_name = 'main_signed_frequency'


    # Add personalized legend for the size of the rectangle
    common_width = bars[0].get_width()
    sorted_heights = sorted(list(set([abs(bar.get_height()) for bar in bars])))
    # print(ax._get_aspect_ratio())
    # legend_rectangles = [mpatches.Rectangle((sorted_heights[-1] - height, 0), height, common_width) for height in sorted_heights]
    # ax.legend(
    #     handles=legend_rectangles,
    #     labels=[f'{int(height)}%' for height in sorted_heights],
    #     loc='lower right',
    # )

    # Add horizontal line at 0
    x_min, x_max = ax.get_xlim()
    ax.hlines(0, x_min, x_max, color='k')

    # Add colorbar
    fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

    show_or_save_plot(plot_name, show)


if __name__ == '__main__':
    # x =[0.385995396446103, 0.354459224712141, 0.337806656992642, 0.217546569145686, 0.00228250945259560, 0.00127876649658226, 0.00113468731616975, 0.000340485790682647, 0.000204625950327110, 0.00537614070678762, 0.00402031528038180, 0.00276905755093206]
    # colors = plt.get_cmap('Greens')(x)
    # print(colors)
    main_plot_signed_frequency(show=True)


