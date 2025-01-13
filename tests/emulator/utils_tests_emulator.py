import numpy as np

from emulator.pysr_emulator.pysr_emulator import PySREmulator
from emulator.pysr_emulator_validated.pysr_emulator_validated import PySREmulatorValidated


def load_pysr_emulator_validated_for_test(**kwargs) -> PySREmulatorValidated:
    return PySREmulatorValidated(niterations=1, **kwargs)


def load_pysr_emulator_for_test(**kwargs) -> PySREmulator:
    return PySREmulator(niterations=1, **kwargs)


def load_X_and_y_for_test() -> tuple[np.ndarray, np.ndarray]:
    X = np.expand_dims(np.arange(100), axis=-1)
    y = X[:, 0] ** 2 - 2 * X[:, 0] + 3
    return X, y


def run_three_main_functions(emulator: PySREmulator):
    X, y = load_X_and_y_for_test()
    emulator.fit(X, y)
    emulator.predict(X)
    emulator.predict(X, index=0)
