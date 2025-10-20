from typing import Optional

from matplotlib import pyplot as plt

from data.utils_dataset.dataset import Dataset
from emulator.emulator import Emulator
from plot.utils_metric.utlis_metric_box import add_metric_box
from plot.by_split.utils_plot_by_split import load_split_name_to_X_and_y_and_y_predicted_and_years, \
    get_ymin_and_ymax
from plot.by_split.utlis_plot_selected_equation import add_equation, get_label, get_true_and_predicted_label
from utils.utils_plot import show_and_save_with_optional_plot_folder


def plot_time_series(emulator: Emulator, dataset: Dataset, show: Optional[bool] = False, plot_folder: Optional[str] = None) -> None:
    """Plot predicted values VS True values (as 2 time series)"""
    split_name_to_X_and_y_and_y_predicted_and_years = load_split_name_to_X_and_y_and_y_predicted_and_years(emulator, dataset.X_train, dataset.y_train, dataset.X_test, 
                                                       dataset.y_test, dataset.years_train, dataset.years_test, dataset.validation_mask)
    ymin, ymax = get_ymin_and_ymax(split_name_to_X_and_y_and_y_predicted_and_years)
    for split_name, (X, y, y_predicted, years) in split_name_to_X_and_y_and_y_predicted_and_years.items():
        ax = plt.gca()
        #  Add grid on Y-axis and two lines
        ax.yaxis.grid()
        y_true_label, y_predicted_label = get_true_and_predicted_label()
        common_kwargs = {'marker': 'o', 'linestyle': ''}
        ax.plot(years, y, label=y_true_label, **common_kwargs)
        ax.plot(years, y_predicted, label=y_predicted_label, **common_kwargs)
        # Annotate equation and metric box
        add_equation(emulator.selected_equation)
        add_metric_box(ax, y, y_predicted, dataset.target_label, split_name, x_and_y_location=(0.05, 0.05))
        #  Add legend and labels
        ax.set_xlabel('Years')
        ax.set_ylabel(get_label(dataset.target_label))
        ax.legend(loc='upper right')
        ax.set_ylim((ymin, ymax))
        show_and_save_with_optional_plot_folder(f'plot_time_series_{split_name}', show, plot_folder)