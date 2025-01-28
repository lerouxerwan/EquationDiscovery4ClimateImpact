import pytest

from emulator.utils_plots.plot_by_rcp.plot_climatological_series import plot_climatological_series
from emulator.utils_plots.plot_by_split.plot_loss_vs_complexity import plot_loss_vs_complexity
from emulator.utils_plots.plot_by_split.plot_scatter import plot_scatter
from emulator.utils_plots.plot_by_split.plot_time_series import plot_time_series
from tests.emulator.utils_tests_emulator import load_climate_impact_emulator_for_test, load_X_and_y_for_test


@pytest.mark.parametrize("plot_function", [plot_loss_vs_complexity, plot_scatter, plot_time_series])
def test_plot_by_split(plot_function):
    emulator = load_climate_impact_emulator_for_test()
    X, y = load_X_and_y_for_test()
    emulator.fit(X, y)
    plot_function(emulator, X, y)

@pytest.mark.parametrize("plot_function", [plot_climatological_series])
def test_plot_by_rcp_for_observed_values(plot_function):
    X, y = load_X_and_y_for_test()
    for nb_historical_years in [0, 20]:
        plot_function(y, nb_historical_years=nb_historical_years)





