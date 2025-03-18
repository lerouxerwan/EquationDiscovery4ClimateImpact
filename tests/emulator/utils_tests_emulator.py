from emulator.pysr_emulator import PySREmulator
from tests.data.utils_tests_dataset import load_X_and_y_for_test


def load_pysr_emulator_for_test(**kwargs) -> PySREmulator:
    return PySREmulator(niterations=1, **kwargs)

def run_three_main_functions_with_one_feature(emulator: PySREmulator):
    X, y = load_X_and_y_for_test()
    run_three_main_functions(emulator, X, y)

def run_three_main_functions(emulator: PySREmulator, X, y):
    emulator.fit(X, y)
    emulator.predict(X)
    emulator.predict(X, index=0)
