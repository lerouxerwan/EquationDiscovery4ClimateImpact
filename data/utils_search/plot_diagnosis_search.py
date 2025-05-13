from collections import Counter

import pandas as pd
from matplotlib import pyplot as plt

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.npp_season_v1 import dataset_npp_season_v1
from data.utils_search.search_experiment import SearchExperiment
from emulator_with_search.utils_cv_results.utils_df_results import RMSE_VALIDATION_COLUMN_NAME, \
    PARAMS_EMULATOR_COLUMN_NAME, SELECTED_FEATURE_INDEXES_COLUMN_NAME
from utils.utils_log import log_info
from utils.utils_plot import show_or_save_plot


def plot_diagnosis_search(dataset: Dataset, search_experiment: SearchExperiment, show: bool = False):
    log_info('Start plot diagnosis search')
    plot_selected_features_best_equation(dataset, search_experiment, show)
    for nb_top_equations in [5, 10, 100]:
        plot_selected_features_top_equations(dataset, search_experiment, nb_top_equations, show)
    # plot_diagnosis_search_1d(search_experiment, show)

def plot_selected_features_top_equations(dataset: Dataset, search_experiment: SearchExperiment,
                                         nb_top_equations: int = 10, show: bool = False):
    """Plot the selected features in the top equations"""
    df = search_experiment.df_cv_results
    if len(df) >= nb_top_equations:
        ax = plt.gca()
        # Gather feature indexes from the top 10 equations
        all_feature_indexes = []
        for selected_feature_indexes in df[SELECTED_FEATURE_INDEXES_COLUMN_NAME].values[:nb_top_equations]:
            all_feature_indexes.extend(selected_feature_indexes)
        c = Counter(all_feature_indexes)
        feature_indexes, occurrences = zip(*sorted(c.items(), key=lambda t: t[1], reverse=True))
        percentages = [100 * o / nb_top_equations for o in occurrences]
        ax.set_ylim(0, 100)
        x_values = range(len(c))
        ax.bar(x_values, percentages, width=0.5)
        ax.set_xticks(x_values)
        xticklabels = [dataset.X_variables_names[feature_index] for feature_index in feature_indexes]
        for selected_feature_index in search_experiment.df_cv_results[SELECTED_FEATURE_INDEXES_COLUMN_NAME].values[0]:
            i = feature_indexes.index(selected_feature_index)
            xticklabels[i] =  '$\\mathbf{' + xticklabels[i] + '}$'
        ax.set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')
        ax.set_ylabel(f'Selection rate in the top {nb_top_equations} equations (%)')
        show_or_save_plot(f'selected_features_top_{nb_top_equations}_equations', show)


def plot_selected_features_best_equation(dataset: Dataset, search_experiment: SearchExperiment, show: bool = False):
    """Plot the selected features in the best equation"""
    selected_feature_indexes = search_experiment.df_cv_results[SELECTED_FEATURE_INDEXES_COLUMN_NAME].values[0]
    for selected_feature_index in selected_feature_indexes:
        ax = plt.gca()
        dataset.plot_values_feature(ax, selected_feature_index)
        show_or_save_plot(f'feature_#{selected_feature_index}', show)

def plot_diagnosis_search_1d(search_experiment: SearchExperiment, show: bool):
    """Plot the variation of RMSE validation for each hyperparameter in the param_grid"""
    df = search_experiment.df_cv_results
    params_list = df[PARAMS_EMULATOR_COLUMN_NAME].to_list()
    metric_name = RMSE_VALIDATION_COLUMN_NAME
    for param_name in search_experiment.get_combinations_of_param_names_in_param_grid(nb_elements=1):
        param_name = param_name[0]
        ax = plt.gca()
        min_loss_list = []
        param_values = [params[param_name] for params in params_list]
        sorted_param_values = sorted(list(set(param_values)))
        for sorted_param_value in sorted_param_values:
            ind = pd.Series(index=df.index, data=[v == sorted_param_value for v in param_values])
            min_loss = df.loc[ind, metric_name].min()
            min_loss_list.append(min_loss)
        ax.plot(sorted_param_values, min_loss_list)
        create_label = lambda s: ' '.join([w.capitalize() for w in s.split('_')])
        ax.set_xlabel(create_label(param_name))
        ax.set_ylabel(create_label(metric_name))
        show_or_save_plot(f"diagnosis_1D_{param_name}", show)

if __name__ == '__main__':
    search_path = ("/home/e23lerou/Documents/EquationDiscovery4ClimateImpact/data/search/82421e3d24e9c7423c3d8163d183862b/b1f603382535a297e302ebec8e7961d4")
    plot_diagnosis_search(dataset_npp_season_v1, SearchExperiment(search_path), show=False)