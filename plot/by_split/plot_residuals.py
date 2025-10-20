import seaborn as sns
from typing import Optional

import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import stats

from data.utils_dataset.dataset import Dataset
from emulator.emulator import Emulator
from plot.utils_metric.utlis_metric_box import add_metric_box
from plot.by_split.utils_plot_by_split import load_split_name_to_X_and_y_and_y_predicted_and_years, get_ymin_and_ymax
from plot.by_split.utlis_plot_selected_equation import get_true_label_and_predicted_label, add_equation, get_unit
from utils.utils_plot import show_and_save_with_optional_plot_folder


def plot_residuals(emulator: Emulator, dataset: Dataset, show: Optional[bool] = False, plot_folder: Optional[str] = None) -> None:
    """Plot the distribution of residuals"""
    split_name_to_X_and_y_and_y_predicted_and_years = load_split_name_to_X_and_y_and_y_predicted_and_years(emulator, dataset.X_train, dataset.y_train, dataset.X_test,
                                                       dataset.y_test, dataset.years_train, dataset.years_test, dataset.validation_mask)
    for split_name, (X, y, y_predicted, years) in split_name_to_X_and_y_and_y_predicted_and_years.items():
        fig, ax = plt.subplots()
        residuals = y_predicted - y
        for bw in [0.3, 0.5, 0.7]:
            sns.kdeplot(residuals, bw=bw, label=f'bandwidth={bw}')
        ax.plot(residuals, [0] * len(residuals), 'k')
        # Add grid and diagonal line
        ax.grid()
        # Annotate equation and metric box
        add_equation(emulator.selected_equation)
        # Add legend and labels
        ax.legend()
        ax.set_xlabel(f'Residuals for {split_name} set ({dataset.y_units[0]})')
        ax.set_ylabel('Density')
        show_and_save_with_optional_plot_folder(f'plot_residuals_{split_name}', show, plot_folder)







