
import numpy as np
import matplotlib.pyplot as plt
from pip._internal.models import index
from scipy.stats import norm

from utils.utils_plot import show_or_save_plot


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

def plot_non_stationary_gaussian(index: int, show: bool = True):

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

    show_or_save_plot("non_stationary_gaussian_{}".format(index), show=show)

if __name__ == '__main__':
    for index in list(range(4))[::-1]:
        plot_non_stationary_gaussian(index, show=False)