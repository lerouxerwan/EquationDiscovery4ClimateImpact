from emulator.climate_impact_emulator import ClimateImpactEmulator
from emulator.utils_plots.plot_selected_equation.plot_scatter import plot_scatter
from emulator.utils_plots.plot_selected_equation.plot_time_series import plot_time_series
from tests.emulator.utils_tests_emulator import load_X_and_y_for_test


def main_example_1d():
    X, y = load_X_and_y_for_test()
    emulator = ClimateImpactEmulator()
    # emulator = ClimateImpactEmulatorWithSearch(n_iter=2)
    emulator.fit(X, y)
    # plot_loss_vs_complexity(emulator, {'train': (X, y)}, show=True)
    plot_scatter(emulator, {'train': (X, y)}, show=True)
    # years = list(range(len(y)))
    # plot_time_series(emulator, {'train': (X, y, years)}, show=True)


if __name__ == '__main__':
    main_example_1d()