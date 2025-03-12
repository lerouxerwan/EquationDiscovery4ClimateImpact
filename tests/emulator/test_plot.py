from emulator.utils_plots.plot_by_rcp.plot_climato import plot_errors_climato, plot_climato
from emulator.utils_plots.plot_by_split.plot_loss_vs_complexity import plot_loss_vs_complexity
from emulator.utils_plots.plot_by_split.plot_scatter import plot_scatter
from emulator.utils_plots.plot_by_split.plot_time_series import plot_time_series
from tests.data.utils_tests_dataset import load_X_and_y_for_test
from tests.emulator.utils_tests_emulator import load_climate_impact_emulator_for_test


def test_plot_with_emulator():
    emulator = load_climate_impact_emulator_for_test()
    X, y = load_X_and_y_for_test()
    emulator.fit(X, y)
    # plot by split and by rcp
    for plot_function in [plot_loss_vs_complexity, plot_scatter, plot_time_series, plot_climato, plot_errors_climato]:
        plot_function(emulator, X, y)


# def test_plot_without_emulator():
#     _, y = load_X_and_y_for_test()
#     for nb_historical_years in [0, 20]:
#         plot_observed_climato(y, nb_historical_years=nb_historical_years)





