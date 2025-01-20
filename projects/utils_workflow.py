from data.dataset.utils_dataset import load_dataset
from emulator.utils_plots.plot_full_diagnosis import plot_full_diagnosis
from emulator.climate_impact_emulator_with_search import ClimateImpactEmulatorWithSearch


def workflow(filename: str, nb_features: int, param_grid: dict[str, list], n_jobs: int, n_iter: int, show: bool) -> None:
    """Workflow that fit an emulator and generate diagnosis plots to assess the quality of this emulator"""
    # Load dataset
    X_train, y_train, X_test, y_test, X_units, y_units, years_train, years_test, variable_names, target_label, index_start_validation = load_dataset(filename)
    # Load emulator
    emulator = ClimateImpactEmulatorWithSearch(select_k_features=nb_features, param_grid=param_grid,
                                               n_jobs=n_jobs, n_iter=n_iter)
    # Fit emulator
    emulator.fit(X_train, y_train, variable_names=variable_names,
                 X_units=X_units, y_units=y_units, index_start_validation=index_start_validation)
    X_train_train, y_train_train = emulator.get_X_and_y(X_train, y_train, validation_set=False)
    X_train_validation, y_train_validation = emulator.get_X_and_y(X_train, y_train, validation_set=True)
    # Generate plots based on the 3 splits
    split_names = ['Train', 'Validation', 'Test']
    X_list = [X_train_train, X_train_validation, X_test]
    y_list = [y_train_train, y_train_validation, y_test]
    years_list = [years_train[~emulator.ind_validation_], years_train[emulator.ind_validation_], years_test]
    split_name_to_X_and_y_and_years = {split_name: (X, y, years)
                             for split_name, X, y, years in zip(split_names, X_list, y_list, years_list)}
    plot_full_diagnosis(emulator, split_name_to_X_and_y_and_years, target_label, show)

