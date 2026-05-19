from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt, patches
from matplotlib.axes import Axes

from emulator.emulator import Emulator
from utils.utils_plot import show_or_save_plot
from utils.utils_run import random_seed


@dataclass
class GaussianToyModel:
    mu_degree: int
    sigma_degree: int
    nb_samples: int = 1000
    stop_sampling: int = 4
    params: Optional[dict] = None

    #####   FIT    ######

    @cached_property
    def emulator(self):
        params = {} if self.params is None else self.params
        emulator = Emulator(gaussian_fit=True, X_variable_names_for_gaussian_fit=['x'],
                            y_variable_name_for_gaussian_fit='y', **params)
        emulator.fit(self.X, self.y, variable_names=['x'])
        return emulator

    #####   PLOT     ######

    def plot(self, show: bool = True, subplot=3):
        """
        Subplot is a number that characterizes the plot that must be shown
        3 means the three plots
        0 means just the samples
        1 means just the ground truth distribution
        2 means just the discovered distribution
        """
        ax = plt.gca()
        # Preprocessing
        nb_samples_for_plot = 20
        x_for_plot = np.linspace(0, self.stop_sampling, num=nb_samples_for_plot)
        X_for_plot = np.expand_dims(x_for_plot, axis=1)
        #  Plot samples
        if subplot in [0, 3]:
            X, y = self.X_and_y
            ax.plot(X[:, 0], y, color='red', linestyle='', marker='o', markersize=1)
        # Plot ground truth values
        if subplot in [1, 3]:
            mu_values = np.array([self.get_mu(x) for x in x_for_plot])
            sigma_values = np.array([self.get_sigma(x) for x in x_for_plot])
            label = 'Ground Truth: $\\mu = {}; log(\\sigma) = {}$'.format(self.get_mu_str(), self.get_log_sigma_str())
            self._plot_gaussian_curve(ax, x_for_plot, mu_values, sigma_values, 'r', label)
        # Plot estimated values
        if subplot in [2, 3]:
            mu_values = self.emulator.get_distri_param(X_for_plot, 'mu')
            sigma_values = self.emulator.get_distri_param(X_for_plot, 'sigma')
            label = 'Discovered: {}'.format(self.emulator.selected_equation)
            self._plot_gaussian_curve(ax, x_for_plot, mu_values, sigma_values, 'blue', label)
        # Postprocessing
        if subplot in [1, 2, 3]:
            ax.legend(loc='upper left')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # Add second legend
        ax_twin = ax.twinx()
        ax_twin.set_xlim(ax.get_xlim())
        ax_twin.set_yticks([])
        legend_handles, legend_labels = [], []
        if subplot in [1, 2, 3]:
            legend_handles += [
                plt.Line2D([0], [0], marker='', linestyle='-', color='k'),
                patches.Patch(facecolor='white', edgecolor='k')]
            legend_labels += ['Average $\\mu$(x)', 'Spread +/- $\\sigma$']
        if subplot in [0, 3]:
            legend_handles += [plt.Line2D([0], [0], marker='o', linestyle='', color='k')]
            legend_labels += ['Training samples']

        ax_twin.legend(legend_handles, legend_labels, loc='lower left', ncol=3)
        plot_name = "toy_model_{}_{}_{}_{}".format(self.mu_degree, self.sigma_degree, self.nb_samples, subplot)
        show_or_save_plot(plot_name=plot_name, show=show)

    @staticmethod
    def _plot_gaussian_curve(ax: Axes, x_values, mu_values, sigma_values, color, label):
        ax.plot(x_values, mu_values, color=color)
        ax.fill_between(x_values, mu_values - sigma_values, mu_values + sigma_values, alpha=0.5, color=color, label=label)

    #####   SAMPLING     ######

    @property
    def X(self):
        return self.X_and_y[0]

    @property
    def y(self):
        return self.X_and_y[1]


    @cached_property
    def X_and_y(self):
        np.random.seed(random_seed)
        samples = []
        x_array = np.linspace(0, self.stop_sampling, num=self.nb_samples)
        for x in x_array:
            mu = self.get_mu(x)
            sigma = self.get_sigma(x)
            sample = np.random.normal(loc=mu, scale=sigma, size=1)[0]
            samples.append(sample)
        X = np.expand_dims(x_array, axis=1)
        y = np.array(samples)
        return X, y

    def get_mu(self, x):
        mu = 1
        if self.mu_degree > 0:
            mu += x
        if self.mu_degree > 1:
            mu += x * x
        if self.mu_degree > 2:
            raise NotImplementedError
        return mu

    def get_mu_str(self):
        mu = "1"
        if self.mu_degree > 0:
            mu += " + x"
        if self.mu_degree > 1:
            mu += " + x^2"
        if self.mu_degree > 2:
            raise NotImplementedError
        return mu


    def get_sigma(self, x):
        log_sigma = 1
        if self.sigma_degree > 0:
            log_sigma += x
        if self.sigma_degree > 1:
            raise NotImplementedError
        return np.exp(log_sigma)

    def get_log_sigma_str(self):
        log_sigma = "1"
        if self.sigma_degree > 0:
            log_sigma += " + x"
        if self.sigma_degree > 1:
            raise NotImplementedError
        return log_sigma
