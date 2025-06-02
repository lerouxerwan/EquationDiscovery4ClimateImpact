from emulator.emulator import Emulator
from emulator.emulator_validated import EmulatorValidated
from emulator.emulator_validated_with_search import EmulatorValidatedWithSearch
from tests.data.utils_tests_dataset import load_X_and_y_for_test, load_X_and_y_and_validation_mask_for_test


def load_pysr_emulator_for_test(**kwargs) -> Emulator:
    return Emulator(niterations=1, **kwargs)

def run_three_main_functions_with_one_feature(emulator: Emulator):
    fit_with_validation_mask = isinstance(emulator, (EmulatorValidated, EmulatorValidatedWithSearch))
    if fit_with_validation_mask:
        X, y, validation_mask = load_X_and_y_and_validation_mask_for_test()
        emulator.fit(X, y, validation_mask)
    else:
        X, y = load_X_and_y_for_test()
        emulator.fit(X, y)
    emulator.predict(X)
    emulator.predict(X, index=0)
