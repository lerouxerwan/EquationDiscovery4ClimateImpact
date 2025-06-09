import os.path as op
from typing import Optional

from data.utils_dataset.dataset import Dataset
from emulator.emulator import Emulator
from emulator.emulator_validated_with_search import EmulatorValidatedWithSearch
from plot.by_rcp.plot_climato import plot_climato, plot_errors_climato
from plot.by_split.plot_loss_vs_complexity import plot_loss_vs_complexity
from plot.by_split.plot_scatter import plot_scatter
from plot.by_split.plot_time_series import plot_time_series
from plot.dataset.plot_selected_features import plot_selected_features
from plot.for_search.plot_diagnosis_search import plot_diagnosis_search
from utils.utils_bash_call import bash_call
from utils.utils_log import log_info
from utils.utils_plot import PLOT_PATH


def plot_diagnosis(emulator: Emulator, dataset:Dataset, show: Optional[bool] = False, plot_folder: Optional[str] = None):
    """Plot diagnosis of this emulator i) by split ii) by rcp iii) for the search"""
    log_info('Start plot diagnosis')
    # Select plot functions
    plot_functions = [plot_selected_features, # plot related to the selected equation
        plot_loss_vs_complexity, plot_scatter, plot_time_series, # plot by split
        plot_climato, plot_errors_climato]  # plot by rcp
    if isinstance(emulator, EmulatorValidatedWithSearch):
        plot_functions.append(plot_diagnosis_search)
    # Run several plot functions
    for plot_function in plot_functions:
        plot_function(emulator, dataset, show, plot_folder)
    # Create two symbolic links between the experiment path in the plot path
    if show is False:
        add_two_symbolic_link(emulator.experiment_.experiment_path, PLOT_PATH)

def add_two_symbolic_link(experiment_path: str, plot_path: str) -> None:
    _add_symbolic_link(experiment_path, plot_path, 'experiment_path')
    _add_symbolic_link(plot_path, experiment_path, 'plot_path')

def _add_symbolic_link(path1: str, path2: str, link_name: str) -> None:
    if not op.exists(op.join(path1, link_name)):
        bash_call(f'ln -s {path1} {op.join(path2, link_name)}')


