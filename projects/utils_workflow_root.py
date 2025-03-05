from emulator.utils_plots.utils_plots import plot_diagnosis
from utils.utils_dataset import load_dataset_dataframe
from emulator_with_search.climate_impact_emulator_with_search import ClimateImpactEmulatorWithSearch


def workflow_root(filename: str, show: bool = False, **params_emulator) -> None:
    """Workflow that fit an emulator and generate diagnosis plots to assess the quality of this emulator"""
    # Load dataset
    (X_train, y_train, X_test, y_test, X_units, y_units, years_train, years_test, rcp_name_train, rcp_name_test,
    variable_names, target_label, nb_historical_years) = load_dataset_dataframe(filename)
    # Fit emulator with search
    emulator = ClimateImpactEmulatorWithSearch(**params_emulator)
    emulator.fit(X_train, y_train, variable_names=variable_names, X_units=X_units, y_units=y_units,
                 index_start_validation=nb_historical_years)
    # Plots
    plot_diagnosis(emulator, X_train, y_train, X_test, y_test, years_train, years_test, rcp_name_train, rcp_name_test, nb_historical_years, target_label, show)




