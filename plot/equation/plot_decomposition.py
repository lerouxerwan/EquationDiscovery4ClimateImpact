from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from sympy import Expr, expand

from data.utils_dataset.dataset import Dataset
from emulator.emulator import Emulator
from utils.utils_plot import show_or_save_plot


def plot_decomposition(emulator: Emulator, dataset:Dataset, show: Optional[bool] = False) -> None:
    for j, expr in enumerate(emulator.selected_expressions):
        _plot_decomposition(expr, dataset.X_train, dataset.X_variable_names, dataset.years_train,
                            f'train_{j}', show)
        _plot_decomposition(expr, dataset.X_test, dataset.X_variable_names, dataset.years_test,
                            f'test_{j}', show)

def _plot_decomposition(expr: Expr, X: ndarray, variable_names: list[str], years: list[int], suffix_plot_name: str,
                        show: Optional[bool] = False):
    # Extract the term of the equation and their percentages through time
    matrix_of_percentages, terms = extract_terms_and_percentages(expr, X, variable_names)
    # Plot the percentage versus time
    ax = plt.gca()
    for term, percentages in zip(terms, matrix_of_percentages):
        ax.plot(years, percentages, label=str(term))
    ax.legend()
    show_or_save_plot(f'percentages_vs_time_for_{suffix_plot_name}', show)
    # Plot the percentage with stacks
    ax = plt.gca()
    ax.stackplot(years, *matrix_of_percentages, labels=[str(term) for term in terms], alpha=0.7)
    ax.legend()
    show_or_save_plot(f'decomposition_for_{suffix_plot_name}', show)


def extract_terms_and_percentages(expr: Expr, X: ndarray, variable_names: list[str]):
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


