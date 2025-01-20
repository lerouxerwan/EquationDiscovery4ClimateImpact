from typing import Any

from matplotlib import pyplot as plt

from emulator.climate_impact_emulator import ClimateImpactEmulator
from emulator.utils_metric.utlis_metric_box import add_metric_box
from emulator.utils_plots.plot_selected_equation.utlis_plot_selected_equation import get_labels, add_equation
from utils.utils_plot import show_or_save_plot


def plot_time_series(emulator: ClimateImpactEmulator, split_name_to_x_and_y_and_years: dict[str, Any],
                     target_label: str = "Target (-)", show: bool = False) -> None:
    """Plot predicted values VS True values (in a scattered way) side by side"""
    for split_name, (X, y, years) in split_name_to_x_and_y_and_years.items():
        ax = plt.gca()
        y_predicted = emulator.predict(X)
        #  Add grid on Y-axis and two lines
        ax.yaxis.grid()
        y_true_label, y_predicted_label = get_labels(target_label, remove_units=True)
        ax.plot(years, y, label=y_true_label)
        ax.plot(years, y_predicted, label=y_predicted_label)
        # Annotate equation and metric box
        add_equation(ax, emulator.selected_expr)
        add_metric_box(ax, y, y_predicted, target_label, split_name.capitalize(), (0.5, 0.1))
        #  Add legend and labels
        ax.set_xlabel('Years')
        ax.set_ylabel(target_label)
        ax.legend(loc='bottom right')
        show_or_save_plot(f'plot_time_series_{split_name}', show)