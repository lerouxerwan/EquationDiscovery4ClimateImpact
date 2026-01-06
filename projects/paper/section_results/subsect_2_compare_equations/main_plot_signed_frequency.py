from typing import OrderedDict, Counter

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, patches
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
        'Greens_centered',
        cmap_original(np.linspace(0.2, 0.8, 256))
    ) # remove extremum colored values
    colors = cmap_new(sorted_average_weight)
    sm = ScalarMappable(cmap=cmap_new, norm=plt.Normalize(vmin=0, vmax=1))
    bars = ax.bar(x_values, y_values, color=colors)
    ax.set_xlabel('Variable names')
    ax.set_xticks(x_values)
    ax.set_ylabel('Contribution to the predicted value')
    ymin, ymax = ax.get_ylim()
    y_tick = max(-ymin, ymax) / 2
    ax.set_yticks([-y_tick, y_tick])
    ax.set_yticklabels(['negative', 'positive'], rotation=90, rotation_mode='anchor', ha='center')
    labels = [label.replace('AnnSea', 'Annual') for label in labels]
    labels = ['$' + label.replace('_', '_{') + '}$' for label in labels]
    ax.set_xticklabels(labels, rotation=45, ha='right', rotation_mode='anchor')
    plot_name = 'main_signed_frequency'

    # Ajout des labels sur chaque barre
    for variable_signed_name, bar in zip(sorted_variable_signed_name, bars):
        text = f'{variable_signed_name_to_number[variable_signed_name]}/{nb_loop}'
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # Position x (centre de la barre)
            bar.get_height() / 2,  # Position y (hauteur de la barre)
            text,  # Texte à afficher (valeur de la barre)
            ha='center',  # Alignement horizontal
            va='center',  # Alignement vertical
            fontsize=8
        )

    # Add horizontal line at 0
    x_min, x_max = ax.get_xlim()
    ax.hlines(0, x_min, x_max, color='k')

    # Add colorbar
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Average relative contribution (%)', rotation=270)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['0%', '50%', '100%'])

    show_or_save_plot(plot_name, show)


if __name__ == '__main__':
    main_plot_signed_frequency(show=False)


