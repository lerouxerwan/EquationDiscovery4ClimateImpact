from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from data.utils_dataset.dataset import Dataset
from emulator.emulator import Emulator
from plot.by_split.utils_plot_by_split import load_split_name_to_X_and_y_and_y_predicted_and_years
from utils.utils_plot import show_or_save_plot


def plot_uncertainty_vs_error(emulator: Emulator, dataset:Dataset, show: Optional[bool] = False):
    """When the predicted mean (mu) is far from the reference value, we want to check if this prediction was deemed uncertain or not"""
    split_name_to_X_and_y_and_y_predicted_and_years = load_split_name_to_X_and_y_and_y_predicted_and_years(emulator, dataset.X_train, dataset.y_train, dataset.X_test,
                                                       dataset.y_test, dataset.years_train, dataset.years_test, dataset.validation_mask)
    for split_name, (X, y, y_predicted, years) in split_name_to_X_and_y_and_y_predicted_and_years.items():
        ax = plt.gca()
        width_uncertainty_interval = [up - low for low, up in emulator.predict_uncertainty_interval(X)]
        rmse_absolute_errors = np.abs(y - y_predicted)
        ax.scatter(rmse_absolute_errors, width_uncertainty_interval)
        model = LinearRegression()
        X_errors = np.expand_dims(rmse_absolute_errors, axis=1)
        model.fit(X_errors, width_uncertainty_interval)
        start, stop = ax.get_xlim()
        error_list = np.expand_dims(np.linspace(start, stop, num=20), axis=1)
        fitted_list = model.predict(error_list)
        ax.plot(error_list[:, 0], fitted_list, color='k')
        ax.set_xlabel('Absolute error')
        ax.set_ylabel('Uncertainty interval (2 x std)')
        show_or_save_plot(f'uncertainty_versus_error', show)

