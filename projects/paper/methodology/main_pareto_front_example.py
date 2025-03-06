import numpy as np

from emulator.pysr_emulator import PySREmulator
from projects.paper.methodology.plot_pareto_front_example import plot_pareto_front_example


def main_example_1d():
    n = 100
    X = np.expand_dims(np.arange(n), axis=-1).astype(float)
    y = X[:, 0] ** 2 + 2 * X[:, 0] + 3
    emulator = PySREmulator(niterations=5, maxsize=9)
    emulator.fit(X, y)
    plot_pareto_front_example(emulator, X, y, show=False)


if __name__ == '__main__':
    main_example_1d()