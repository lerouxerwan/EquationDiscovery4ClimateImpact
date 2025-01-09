from pysr_emulator.pysr_emulator import PySREmulator
from tests.pysr_emulator.utils_tests_pysr_emulator import load_X_and_y_for_test


def main_example_1d():
    X, y = load_X_and_y_for_test()
    emulator = PySREmulator()
    emulator.fit(X, y)
    y_predict = emulator.predict(X)
    print(y_predict)

if __name__ == '__main__':
    main_example_1d()