from dataclasses import dataclass
from functools import cached_property

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from emulator.emulator import Emulator
from utils.utils_run import random_seed


@dataclass
class GaussianToyModel:
    mu_degree: int
    sigma_degree: int
    nb_samples: int = 1000
    stop_sampling: int = 4

    #####   FIT    ######

    @cached_property
    def emulator(self):
        emulator = Emulator(gaussian_fit=True, X_variable_names_for_gaussian_fit=['x'],
                            y_variable_name_for_gaussian_fit='y')
        emulator.fit(self.X, self.y, variable_names=['x'])
        return emulator

    #####   PLOT     ######

    def plot(self):
        ax = plt.gca()
        # Preprocessing
        nb_samples_for_plot = 20
        x_for_plot = np.linspace(0, self.stop_sampling, num=nb_samples_for_plot)
        X_for_plot = np.expand_dims(x_for_plot, axis=1)
        # Plot ground truth values
        mu_values = np.array([self.get_mu(x) for x in x_for_plot])
        sigma_values = np.array([self.get_sigma(x) for x in x_for_plot])
        self._plot_gaussian_curve(ax, x_for_plot, mu_values, sigma_values, 'r', 'reference')
        # Plot estimated values
        mu_values = self.emulator.get_distri_param(X_for_plot, 'mu')
        sigma_values = self.emulator.get_distri_param(X_for_plot, 'sigma')
        self._plot_gaussian_curve(ax, x_for_plot, mu_values, sigma_values, 'blue', 'fit')
        # Postprocessing
        ax.legend()
        ax.set_xlabel('x')
        plt.show()

    @staticmethod
    def _plot_gaussian_curve(ax: Axes, x_values, mu_values, sigma_values, color, label):
        ax.plot(x_values, mu_values, color=color, label=label)
        ax.fill_between(x_values, mu_values - sigma_values, mu_values + sigma_values, alpha=0.5, color=color)

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


    def get_sigma(self, x):
        sigma = 1
        if self.sigma_degree > 0:
            sigma += x
        if self.sigma_degree > 1:
            raise NotImplementedError
        return sigma