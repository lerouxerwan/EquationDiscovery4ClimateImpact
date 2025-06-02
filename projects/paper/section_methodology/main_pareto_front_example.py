import numpy as np
from matplotlib import pyplot as plt

from emulator.emulator import Emulator
from projects.paper.section_methodology.plot_pareto_front_example import plot_pareto_front_example
from utils.utils_plot import show_or_save_plot

n = 100


def main_example_1d(show: bool):
    X, y = get_data_example_1d()
    emulator = Emulator(niterations=5, maxsize=9)
    emulator.fit(X, y)
    plot_pareto_front_example(emulator, X, y, show=show)


def get_data_example_1d():
    X = np.expand_dims(np.arange(n), axis=-1).astype(float)
    y = X[:, 0] ** 2 + 2 * X[:, 0] + 3
    return X, y

def main_pareto_front_example_1d(show: bool):
    X, y = get_data_example_1d()
    ax = plt.gca()
    equation_str = '$x^2 + 2 \\times x + 3$'
    ax.plot(X[:, 0], y, label=('%s' % equation_str), linestyle='', marker='o')
    fontsize = 20
    ax.set_xlabel('x', fontsize=fontsize)
    ax.set_ylabel(equation_str, fontsize=fontsize)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    show_or_save_plot(f'data_generated_example', show)




if __name__ == '__main__':
    main_example_1d(show=True)
    # main_pareto_front_example_1d(show=True)