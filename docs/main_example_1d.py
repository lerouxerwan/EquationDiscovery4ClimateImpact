from emulator.pysr_emulator import PySREmulator
from emulator.utils_plots.plot_by_split.plot_loss_vs_complexity import plot_loss_vs_complexity
from emulator.utils_plots.plot_by_split.plot_scatter import plot_scatter
from emulator.utils_plots.plot_by_split.plot_time_series import plot_time_series
from tests.utils_tests_dataset import load_X_and_y_for_test


def main_example_1d():
    X, y = load_X_and_y_for_test()
    # plot_observed_climato(y, show=True)
    emulator = PySREmulator(niterations=5, maxsize=10)
    emulator.fit(X, y)
    # plot_predicted_climato(emulator, X, show=True)
    for plot in [plot_loss_vs_complexity, plot_scatter, plot_time_series][1:2]:
        plot(emulator, X, y, show=True)


if __name__ == '__main__':
    main_example_1d()