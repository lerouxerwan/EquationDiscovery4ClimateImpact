from data.dataset.utils_dataset import load_dataset
from emulator.pysr_emulator_validated.pysr_emulator_validated import PySREmulatorValidated


def workflow(filename: str, nb_features: int, param_grid: dict[str, list], n_jobs: int, n_iter: int):
    X_train, y_train, X_test, y_test, X_units, y_units, variable_names, index_start_validation = load_dataset(filename)
    # Load emulator
    emulator = PySREmulatorValidated(select_k_features=nb_features, param_grid=param_grid,
                                     n_jobs=n_jobs, n_iter=n_iter)
    # Fit emulator
    emulator.fit(X_train, y_train, variable_names=variable_names,
                 X_units=X_units, y_units=y_units, index_start_validation=index_start_validation)
    # Predict with emulator
    y_train_predicted = emulator.predict(X_train)
    y_test_predicted = emulator.predict(X_test)
    return None
