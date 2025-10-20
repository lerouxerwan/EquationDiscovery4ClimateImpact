import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from emulator.emulator import Emulator
from plot.by_split.plot_loss_vs_complexity import plot_loss_vs_complexity
from plot.plot_diagnosis import plot_diagnosis
from plot.workflow import fit
from utils.utils_run import random_seed

def get_mu(x):
    return x * x + x - 1

def get_sigma(x):
    return 1

def get_X_y_from_normal(nb_samples: int):
    np.random.seed(random_seed)
    samples = []
    x_array = np.linspace(0, 1, num=nb_samples)
    for x in x_array:
        mu = get_mu(x)
        sigma = get_sigma(x)
        sample = np.random.normal(loc=mu, scale=sigma, size=1)[0]
        samples.append(sample)
    X = np.expand_dims(x_array, axis=1)
    y = np.array(samples)
    return X, y


def fit_gaussian(X, y) -> Emulator:
    emulator = Emulator(gaussian_fit=True, X_variable_names_for_gaussian_fit=['x'], y_variable_name_for_gaussian_fit='y')
    emulator.fit(X, y, variable_names=['x'])
    return emulator

def plot_gaussian_curve(ax: Axes, x_values, mu_values, sigma_values, color, label):
    ax.plot(x_values, mu_values, color=color, label=label)
    ax.fill_between(x_values, mu_values - sigma_values, mu_values + sigma_values, alpha=0.5, color=color)


if __name__ == '__main__':
    X, y = get_X_y_from_normal(nb_samples=1000)
    emulator = fit_gaussian(X, y)
    for complexity, equation in zip(emulator.complexity_list, emulator.equation_list):
        print(complexity, equation)
    print(emulator.selected_equation)

    nb_samples_for_plot = 20
    x_for_plot = np.linspace(0, 1, num=nb_samples_for_plot)
    X_for_plot = np.expand_dims(x_for_plot, axis=1)
    # Plot
    ax = plt.gca()
    mu_values = np.array([get_mu(x) for x in x_for_plot])
    sigma_values = np.array([get_sigma(x) for x in x_for_plot])
    plot_gaussian_curve(ax, x_for_plot, mu_values, sigma_values, 'r', 'reference')
    plot_gaussian_curve(ax, x_for_plot, emulator.get_distri_param(X_for_plot, 'mu'),
                        emulator.get_distri_param(X_for_plot, 'sigma'), 'blue', 'fit')
    ax.legend()
    plt.show()



