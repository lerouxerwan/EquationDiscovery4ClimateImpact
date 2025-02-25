from data.utils_dataset import load_dataset_dataframe
from emulator.climate_impact_emulator_with_search import ClimateImpactEmulatorWithSearch
from emulator.utils_plots.plot_by_rcp.plot_climato import plot_errors_climato, plot_climato
from emulator.utils_plots.plot_by_split.plot_loss_vs_complexity import plot_loss_vs_complexity
from emulator.utils_plots.plot_by_split.plot_scatter import plot_scatter
from emulator.utils_plots.plot_by_split.plot_time_series import plot_time_series


def workflow(filename: str, show: bool = False, **params_emulator) -> None:
    """Workflow that fit an emulator and generate diagnosis plots to assess the quality of this emulator"""
    # Load dataset
    (X_train, y_train, X_test, y_test, X_units, y_units, years_train, years_test, rcp_name_train, rcp_name_test,
    variable_names, target_label, nb_historical_years) = load_dataset_dataframe(filename)
    # Fit emulator with search
    emulator = ClimateImpactEmulatorWithSearch(**params_emulator)
    emulator.fit(X_train, y_train, variable_names=variable_names, X_units=X_units, y_units=y_units,
                 index_start_validation=nb_historical_years)
    # Plots
    # Plot diagnosis of this emulator by split
    for plot_function in [plot_loss_vs_complexity, plot_scatter, plot_time_series]:
        plot_function(emulator, X_train, y_train, X_test, y_test, years_train, years_test, rcp_name_train,
                      rcp_name_test, nb_historical_years, target_label, show)
    # # Plot diagnosis of this emulator by rcp
    for plot_function in [plot_climato, plot_errors_climato]:
        plot_function(emulator, X_train, y_train, X_test, y_test, years_train, years_test, rcp_name_train,
                            rcp_name_test, nb_historical_years, target_label, show)




