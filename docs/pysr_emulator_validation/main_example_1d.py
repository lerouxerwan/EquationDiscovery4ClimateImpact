from emulator.pysr_emulator_validated.pysr_emulator_validated import PySREmulatorValidated
from tests.emulator.utils_tests_emulator import load_X_and_y_for_test


def main_example_1d():
    X, y = load_X_and_y_for_test()
    emulator = PySREmulatorValidated(param_grid={'niterations': [1, 2]})
    emulator.fit(X, y)
    y_predict = emulator.predict(X)
    print(y_predict)

if __name__ == '__main__':
    main_example_1d()