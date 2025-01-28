import numpy as np

from emulator.climate_impact_emulator import ClimateImpactEmulator
from projects.paper.methodology.plot_pareto_front_example import plot_pareto_front_example
from tests.emulator.utils_tests_emulator import load_X_and_y_for_test


def main_example_1d():
    n = 100
    X = np.expand_dims(np.arange(n), axis=-1).astype(float)
    y = X[:, 0] ** 2 + 2 * X[:, 0] + 3
    emulator = ClimateImpactEmulator(niterations=5, maxsize=9)
    emulator.fit(X, y)
    plot_pareto_front_example(emulator, X, y, show=False)


if __name__ == '__main__':
    main_example_1d()