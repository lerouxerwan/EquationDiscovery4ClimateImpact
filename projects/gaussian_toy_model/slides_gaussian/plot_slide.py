import random

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal
from pip._internal.models import index
from scipy.stats import norm

from utils.utils_plot import show_or_save_plot
from utils.utils_run import random_seed


def loc_and_sigma(t, index: int):
    default_sigma = 0.5
    if index == 0:
        return 0, default_sigma
    elif index == 1:
        return t, default_sigma
    elif index == 2:
        return 0, default_sigma + t * 0.25
    elif index == 3:
        return t, default_sigma + t * 0.25
    else:
        raise NotImplementedError

def plot_non_stationary_gaussian_samples(index: int, show: bool = True):
    # Set random seed
    random.seed(random_seed)
    # Range of values
    t_values = np.linspace(0, 3, 4)  # 5 pas de temps : 0, 2.5, 5, 7.5, 10

    ax = plt.gca()
    # Pour chaque t, tracer la densité gaussienne verticalement, centrée sur t
    for t in t_values:
        loc, sigma = loc_and_sigma(t, index)
        values = normal(loc, sigma, size=1)
        ax.plot([t], values, marker='o', label="$\mu(t) = {}$\n$\sigma(t) = {}$".format(loc, sigma))

    ymin, ymax = ax.get_ylim()
    ax.set_ylim((ymin, ymax * 1.2))
    ax.legend(ncol=4, prop={'size': 10.3}, loc='upper center')
    ax.set_xlabel("$t$")
    ax.set_xticks(t_values)

    show_or_save_plot("non_stationary_gaussian_samples_{}".format(index), show=show)


def plot_non_stationary_gaussian_distribution(index: int, show: bool = True):

    # Range of values
    t_values = np.linspace(0, 3, 4)  # 5 pas de temps : 0, 2.5, 5, 7.5, 10
    x_range = np.linspace(-7.5, 7.5, 1000)  # Plage pour x

    ax = plt.gca()
    # Pour chaque t, tracer la densité gaussienne verticalement, centrée sur t
    for t in t_values:
        loc, sigma = loc_and_sigma(t, index)
        pdf = norm.pdf(x_range, loc=loc, scale=sigma)
        ax.plot(t + pdf, x_range, label="$\mu(t) = {}$\n$\sigma(t) = {}$".format(loc, sigma))

    ax.legend(ncol=4, prop={'size': 10.3})
    ax.set_xlabel("$t$")
    ax.set_xticks(t_values)

    show_or_save_plot("non_stationary_gaussian_distribution_{}".format(index), show=show)

if __name__ == '__main__':
    for index in list(range(4))[::-1]:
        plot_non_stationary_gaussian_samples(index, show=False)