from typing import Optional

import numpy as np

from data.utils_dataset.dataset import Dataset
from emulator.emulator import Emulator
from plot.by_rcp.plot_climato import plot_errors_climato, _plot_errors_climato, _plot_climato
from plot.by_split.utlis_plot_selected_equation import get_true_label_and_predicted_label
from utils.utils_plot import get_subplots, compute_axis_lim, show_and_save_with_optional_plot_folder


def plot_climato_side_by_side(emulator: Emulator, dataset: Dataset, show: Optional[bool] = False, plot_folder: Optional[str] = None):
    fig, axs = get_subplots(2, 2, sharex=True, sharey=False)

    # First row
    y_train_predicted = emulator.predict(dataset.X_train)
    y_test_predicted = None if dataset.X_test is None else emulator.predict(dataset.X_test)
    y_values = np.concat([y for y in [dataset.y_train, dataset.y_test, y_train_predicted, y_test_predicted] if y is not None])
    ymin_and_ymax = compute_axis_lim(y_values)
    true_target_label, predicted_target_label = get_true_label_and_predicted_label(dataset.target_label)
    _plot_climato(dataset.y_train, dataset.y_test, dataset.years_train, dataset.years_test, dataset.rcp_name_train, dataset.rcp_name_test,
                  dataset.validation_mask, true_target_label, show, ymin_and_ymax, plot_folder, axs[0, 0])
    _plot_climato(y_train_predicted, y_test_predicted, dataset.years_train, dataset.years_test, dataset.rcp_name_train, dataset.rcp_name_test,
                  dataset.validation_mask, predicted_target_label, show, ymin_and_ymax, plot_folder, axs[0, 1])

    # Second row
    loc = 'upper left'
    _plot_errors_climato(emulator, dataset, show, False, plot_folder, axs[1, 0], loc=loc)
    _plot_errors_climato(emulator, dataset, show, True, plot_folder, axs[1, 1], loc=loc)


    show_and_save_with_optional_plot_folder(f'climatological_series_side_by_side', show, plot_folder)
