from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from sympy import Expr, expand

from data.utils_dataset.dataset import Dataset
from emulator.emulator import Emulator
from utils.utils_plot import show_and_save_with_optional_plot_folder


def plot_decomposition(emulator: Emulator, dataset:Dataset, show: Optional[bool] = False,
                            plot_folder: Optional[str] = None) -> None:
    _plot_decomposition(emulator.selected_expr, dataset.X_train, dataset.X_variable_names, dataset.years_train,
                        'train', show, plot_folder)
    _plot_decomposition(emulator.selected_expr, dataset.X_test, dataset.X_variable_names, dataset.years_test,
                        'test', show, plot_folder)

def _plot_decomposition(expr: Expr, X: np.ndarray, variable_names: list[str], years: list[int], split_name: str,
                        show: Optional[bool] = False, plot_folder: Optional[str] = None):
    # Extract the term of the equation and their percentages through time
    matrix_of_percentages, terms = extract_terms_and_percentages(expr, X, variable_names)
    # Plot the percentage versus time
    ax = plt.gca()
    for term, percentages in zip(terms, matrix_of_percentages):
        ax.plot(years, percentages, label=str(term))
    ax.legend()
    show_and_save_with_optional_plot_folder(f'percentages_vs_time_for_{split_name}', show, plot_folder)
    # Plot the percentage with stacks
    ax = plt.gca()
    ax.stackplot(years, *matrix_of_percentages, labels=[str(term) for term in terms], alpha=0.7)
    ax.legend()
    show_and_save_with_optional_plot_folder(f'decomposition_for_{split_name}', show, plot_folder)


def extract_terms_and_percentages(expr: Expr, X: np.ndarray, variable_names: list[str]):
    #  Extract terms
    terms = expand(expr).args
    #  Extract list of percentages
    matrix_of_percentages = []
    for x in X:
        subs_dict = dict(zip(variable_names, x))
        absolute_weights = np.array([abs(term.subs(subs_dict)) for term in terms])
        percentages = 100 * absolute_weights / sum(absolute_weights)
        matrix_of_percentages.append(percentages)
    matrix_of_percentages = np.transpose(np.array(matrix_of_percentages).astype(int))
    return matrix_of_percentages, terms


