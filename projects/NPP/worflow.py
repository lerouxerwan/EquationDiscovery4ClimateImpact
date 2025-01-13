from emulator.pysr_emulator.pysr_emulator import PySREmulator
from projects.NPP.load_data import load_data


def fit_and_predict():
    X_train, y_train, X_test, y_test, X_units, y_units, variable_names, _ = load_data()
    # Load emulator
    emulator = PySREmulator(select_k_features=5)
    # Fit emulator
    emulator.fit(X_train, y_train, variable_names=variable_names, X_units=X_units, y_units=y_units)
    # Predict with emulator
    emulator.predict(X_test)

if __name__ == '__main__':
    fit_and_predict()


