from typing import Optional

import numpy as np

from emulator.pysr_emulator import PySREmulator
from emulator.utils_plots.plot_by_rcp.plot_climato import plot_climato, plot_errors_climato
from emulator.utils_plots.plot_by_split.plot_loss_vs_complexity import plot_loss_vs_complexity
from emulator.utils_plots.plot_by_split.plot_scatter import plot_scatter
from emulator.utils_plots.plot_by_split.plot_time_series import plot_time_series
from utils.utils_log import log_info


def plot_diagnosis_fit(emulator: PySREmulator, X_train: np.ndarray,
                       y_train: np.ndarray,
                       validation_mask: np.ndarray[bool],
                       X_test: Optional[np.ndarray]=None,
                       y_test: Optional[np.ndarray]=None,
                       years_train: Optional[np.ndarray]=None, years_test: Optional[np.ndarray]=None,
                       rcp_name_train: str= 'RCP85', rcp_name_test: Optional[str]=None,
                       target_label: str = "Target (-)", show: Optional[bool] = False):
    log_info('Start plot diagnosis fit')
    # Plot diagnosis of this emulator by split and by rcp
    for plot_function in [plot_loss_vs_complexity, plot_scatter, plot_time_series, plot_climato, plot_errors_climato]:
        plot_function(emulator, X_train, y_train, validation_mask, X_test, y_test, years_train, years_test, rcp_name_train,
                      rcp_name_test, target_label, show)
