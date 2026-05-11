import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from emulator.emulator import Emulator
from plot.by_split.plot_loss_vs_complexity import plot_loss_vs_complexity
from utils.utils_run import random_seed

STOP = 5


def get_mu(x):
    # return x * x + x - 1
    return x * x + x - 1

def get_sigma(x):
    return np.exp(0)
    # return np.exp(x)

def get_X_y_from_normal(nb_samples: int):
    np.random.seed(random_seed)
    samples = []
    x_array = np.linspace(0, STOP, num=nb_samples)
    for x in x_array:
        mu = get_mu(x)
        sigma = get_sigma(x)
        sample = np.random.normal(loc=mu, scale=sigma, size=1)[0]
        # sample = np.random.gumbel(loc=mu, scale=sigma, size=1)[0]
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


def main_plot_toy_model():
    X, y = get_X_y_from_normal(nb_samples=1000)
    emulator = fit_gaussian(X, y)
    for complexity, equation in zip(emulator.complexity_list, emulator.equation_list):
        print(complexity, equation)
    print(emulator.selected_equation)

    nb_samples_for_plot = 20
    x_for_plot = np.linspace(0, STOP, num=nb_samples_for_plot)
    X_for_plot = np.expand_dims(x_for_plot, axis=1)

    # Plots
    plot_two_curves(X_for_plot, emulator, x_for_plot)


def plot_two_curves(X_for_plot, emulator: Emulator, x_for_plot):
    ax = plt.gca()
    mu_values = np.array([get_mu(x) for x in x_for_plot])
    sigma_values = np.array([get_sigma(x) for x in x_for_plot])
    plot_gaussian_curve(ax, x_for_plot, mu_values, sigma_values, 'r', 'reference')
    plot_gaussian_curve(ax, x_for_plot, emulator.get_distri_param(X_for_plot, 'mu'),
                        emulator.get_distri_param(X_for_plot, 'sigma'), 'blue', 'fit')
    ax.legend()
    ax.set_xlabel('x')
    plt.show()



if __name__ == '__main__':
    main_plot_toy_model()




