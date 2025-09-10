import os.path as op
from typing import Optional

from data.utils_dataset.dataset import Dataset
from emulator.emulator import Emulator
from emulator.emulator_with_search import EmulatorWithSearch
from plot.by_rcp.plot_climato import plot_climato, plot_errors_climato
from plot.by_rcp.plot_climato_side_by_side import plot_climato_side_by_side
from plot.by_split.plot_loss_vs_complexity import plot_loss_vs_complexity
from plot.by_split.plot_residuals import plot_residuals
from plot.by_split.plot_scatter import plot_scatter, plot_scatter_side_by_side
from plot.by_split.plot_time_series import plot_time_series
from plot.dataset.plot_selected_features import plot_selected_features
from plot.equation.plot_decomposition import plot_decomposition
from plot.for_search.plot_diagnosis_search import plot_diagnosis_search
from utils.utils_bash_call import bash_call
from utils.utils_log import log_info
from utils.utils_plot import PLOT_PATH


def plot_diagnosis(emulator: Emulator, dataset:Dataset, show: Optional[bool] = False, plot_folder: Optional[str] = None):
    """Plot diagnosis of this emulator i) by split ii) by rcp iii) for the search"""
    log_info('Start plot diagnosis')
    # Select plot functions
    plot_functions = [
        plot_selected_features, # plot related to the selected equation
        plot_loss_vs_complexity, plot_scatter, plot_residuals, plot_time_series, # plot by split
        plot_climato, plot_errors_climato, plot_climato_side_by_side, # plot by rcp
        plot_scatter_side_by_side, # plot side by side
        plot_decomposition, # plot a decomposition for the selected equation
    ]
    if isinstance(emulator, EmulatorWithSearch):
        plot_functions.append(plot_diagnosis_search)
    # Run several plot functions
    for plot_function in plot_functions:
        plot_function(emulator, dataset, show, plot_folder)
    # Create two symbolic links between the run_directory in the plot path
    if show is False:
        add_two_symbolic_link(emulator.run_.run_directory, PLOT_PATH)

def add_two_symbolic_link(run_directory: str, plot_path: str) -> None:
    _add_symbolic_link(run_directory, op.join(plot_path, 'run_directory'))
    _add_symbolic_link(plot_path, op.join(run_directory, 'plot_path'))

def _add_symbolic_link(path1: str, link_path: str) -> None:
    if not op.exists(link_path):
        bash_call(f'ln -s {path1} {link_path}')


