from data.dataset.utils_dataset import load_dataset
from emulator.climate_impact_emulator_with_search.climate_impact_emulator_with_search import ClimateImpactEmulatorWithSearch


def workflow(filename: str, nb_features: int, param_grid: dict[str, list], n_jobs: int, n_iter: int):
    X_train, y_train, X_test, y_test, X_units, y_units, variable_names, index_start_validation = load_dataset(filename)
    # Load emulator
    emulator = ClimateImpactEmulatorWithSearch(select_k_features=nb_features, param_grid=param_grid,
                                               n_jobs=n_jobs, n_iter=n_iter)
    # Fit emulator
    emulator.fit(X_train, y_train, variable_names=variable_names,
                 X_units=X_units, y_units=y_units, index_start_validation=index_start_validation)
    X_train_train, y_train_train = emulator.get_X_and_y(X_train, y_train, validation_set=False)
    X_train_validation, y_train_validation = emulator.get_X_and_y(X_train, y_train, validation_set=True)
    # Predict and plot
    labels = ['Train', 'Validation', 'Test']
    X_list = [X_train_train, X_train_validation, X_test]
    y_list = [y_train_train, y_train_validation, y_test]
    for label, X, y in zip(labels, X_list, y_list):
        y_predicted = emulator.predict(X)
        # Compute errors and add plots
    return None
