from emulator.climate_impact_emulator import ClimateImpactEmulator
from emulator.utils_plots.plot_by_split.plot_loss_vs_complexity import plot_loss_vs_complexity
from emulator.utils_plots.plot_by_split.plot_scatter import plot_scatter
from emulator.utils_plots.plot_by_split.plot_time_series import plot_time_series
from tests.emulator.utils_tests_emulator import load_X_and_y_for_test


def main_example_1d():
    X, y = load_X_and_y_for_test()
    emulator = ClimateImpactEmulator(niterations=5, maxsize=10)
    # emulator = ClimateImpactEmulatorWithSearch(n_iter=2)
    emulator.fit(X, y)
    for plot in [plot_loss_vs_complexity, plot_scatter, plot_time_series][-1:]:
        plot(emulator, X, y, show=True)


if __name__ == '__main__':
    main_example_1d()