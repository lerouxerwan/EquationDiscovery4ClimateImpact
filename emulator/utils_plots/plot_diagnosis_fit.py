from typing import Optional

import numpy as np

from data.utils_dataset.dataset import Dataset
from data.utils_search.plot_diagnosis_search import plot_diagnosis_search
from emulator.pysr_emulator import PySREmulator
from emulator.utils_plots.plot_by_rcp.plot_climato import plot_climato, plot_errors_climato
from emulator.utils_plots.plot_by_split.plot_loss_vs_complexity import plot_loss_vs_complexity
from emulator.utils_plots.plot_by_split.plot_scatter import plot_scatter
from emulator.utils_plots.plot_by_split.plot_time_series import plot_time_series
from utils.utils_log import log_info



def plot_diagnosis_fit(emulator: PySREmulator, dataset:Dataset, show: Optional[bool] = False, plot_folder: Optional[str] = None):
    log_info('Start plot diagnosis')
    # Plot diagnosis of this emulator by split and by rcp
    for plot_function in [plot_loss_vs_complexity, plot_scatter, plot_time_series,
                          plot_climato, plot_errors_climato,
                          plot_diagnosis_search]:
        plot_function(emulator, dataset, show, plot_folder)
