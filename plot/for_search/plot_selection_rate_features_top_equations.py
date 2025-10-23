from collections import Counter

from matplotlib import pyplot as plt

from data.utils_dataset.dataset import Dataset
from emulator.emulator_with_search import EmulatorWithSearch
from emulator.utils_hyperparameter_search.utils_column_names import VARIABLE_NAMES_COLUMN_NAME
from plot.dataset.plot_selected_features import get_selected_feature_indexes
from utils.utils_plot import show_or_save_plot


def plot_selection_rate_features_top_equations(emulator: EmulatorWithSearch, dataset: Dataset,
                                               nb_top_equations: int = 10, show: bool = False):
    """Plot the selected features for top equations"""
    df = emulator.run_.df_cv_results.copy()
    if len(df) >= nb_top_equations:
        ax = plt.gca()
        # Gather feature indexes from the top 10 equations
        all_feature_indexes = []
        for selected_variable_names in df[VARIABLE_NAMES_COLUMN_NAME].values[:nb_top_equations]:
            all_feature_indexes.extend(get_selected_feature_indexes(selected_variable_names, dataset.X_variable_names))
        c = Counter(all_feature_indexes)
        feature_indexes, occurrences = zip(*sorted(c.items(), key=lambda t: t[1], reverse=True))
        percentages = [100 * o / nb_top_equations for o in occurrences]
        ax.set_ylim(0, 100)
        x_values = range(len(c))
        ax.bar(x_values, percentages, width=0.5)
        ax.set_xticks(x_values)
        xticklabels = [dataset.X_variable_names[feature_index] for feature_index in feature_indexes]
        xticklabels = ['$' + label.replace('_', '_{') + '}$' for label in xticklabels]
        for selected_feature_index in all_feature_indexes[0]:
            i = feature_indexes.index(selected_feature_index)
            xticklabels[i] =  '$\\mathbf{' + xticklabels[i][1:-1] + '}$'
        ax.set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')
        ax.set_ylabel(f'Selection rate for top {nb_top_equations} equations (%)')
        show_or_save_plot(f'selected_features_top_{nb_top_equations}_equations', show)

