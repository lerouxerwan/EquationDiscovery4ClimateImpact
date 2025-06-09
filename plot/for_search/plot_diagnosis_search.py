from typing import Optional

from data.utils_dataset.dataset import Dataset
from emulator.emulator import Emulator
from plot.for_search.plot_search_1d import plot_diagnosis_search_1d
from plot.for_search.plot_selection_rate_features_top_equations import plot_selection_rate_features_top_equations


def plot_diagnosis_search(emulator: Emulator, dataset:Dataset, show: Optional[bool] = False,
                          plot_folder: Optional[str] = None):
    for nb_top_equations in [5, 10, 20]:
        plot_selection_rate_features_top_equations(dataset, emulator.experiment_, nb_top_equations, show, plot_folder)
    plot_diagnosis_search_1d(emulator.experiment_, show, plot_folder)


