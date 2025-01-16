from data.dataset.utils_dataset import load_dataset
from emulator.climate_impact_emulator.utils_plot import generate_plots
from emulator.climate_impact_emulator_with_search.climate_impact_emulator_with_search import ClimateImpactEmulatorWithSearch


def workflow(filename: str, nb_features: int, param_grid: dict[str, list], n_jobs: int, n_iter: int, show: bool) -> None:
    """Workflow that fit an emulator and generate diagnosis plots to assess the quality of this emulator"""
    # Load dataset
    X_train, y_train, X_test, y_test, X_units, y_units, variable_names, index_start_validation = load_dataset(filename)
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
    split_name_to_x_and_y = {split_name: (X, y) for X, y, split_name in zip(X_list, y_list, split_names)}
    generate_plots(emulator, split_name_to_x_and_y, show)

