from emulator.climate_impact_emulator import ClimateImpactEmulator
from emulator.utils_plots.plot_loss_vs_complexity import plot_loss_vs_complexity
from emulator.utils_plots.plot_scatter import plot_scatter
from tests.emulator.utils_tests_emulator import load_X_and_y_for_1D_test


def main_example_1d():
    X, y = load_X_and_y_for_1D_test()
    emulator = ClimateImpactEmulator()
    # emulator = ClimateImpactEmulatorWithSearch(param_grid={'niterations': [10, 20]}, n_iter=2)
    emulator.fit(X, y)
    # plot_loss_vs_complexity(emulator, {'train': (X, y)}, show=True)
    plot_scatter(emulator, {'train': (X, y)}, show=True)


if __name__ == '__main__':
    main_example_1d()