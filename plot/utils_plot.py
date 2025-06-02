from typing import Optional

from data.utils_dataset.dataset import Dataset
from plot.for_search.plot_diagnosis_search import plot_diagnosis_search
from emulator.emulator import Emulator
from plot.by_rcp.plot_climato import plot_climato, plot_errors_climato
from plot.by_split.plot_loss_vs_complexity import plot_loss_vs_complexity
from plot.by_split.plot_scatter import plot_scatter
from plot.by_split.plot_time_series import plot_time_series
from utils.utils_log import log_info



def plot_diagnosis(emulator: Emulator, dataset:Dataset, show: Optional[bool] = False, plot_folder: Optional[str] = None):
    """Plot diagnosis of this emulator i) by split ii) by rcp iii) for the search"""
    log_info('Start plot diagnosis')
    for plot_function in [
        plot_loss_vs_complexity, plot_scatter, plot_time_series, # plot by split
        plot_climato, plot_errors_climato, # plot by rcp
        plot_diagnosis_search # plot for search
    ]:
        plot_function(emulator, dataset, show, plot_folder)
