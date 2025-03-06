import numpy as np
from scipy.stats import norm

from emulator.pysr_emulator import PySREmulator
from emulator_with_search.pysr_emulator_with_search import PySREmulatorWithSearch
from utils.utils_run import random_seed


def load_climate_impact_emulator_for_test(**kwargs) -> PySREmulator:
    return PySREmulator(niterations=1, **kwargs)

def load_climate_impact_emulator_with_search_for_test(**kwargs) -> PySREmulatorWithSearch:
    return PySREmulatorWithSearch(niterations=1, **kwargs)



def load_X_and_y_for_test(nb_features=1) -> tuple[np.ndarray, np.ndarray]:
    n = 100
    X = np.expand_dims(np.arange(n), axis=-1).astype(float)
    y = X[:, 0] ** 2 - 2 * X[:, 0] + 3
    y += norm.rvs(loc=0, scale=1, size=n, random_state=random_seed)
    if nb_features > 1:
        X = np.repeat(X, repeats=nb_features, axis=1)
    return X, y

def run_three_main_functions_with_one_feature(emulator: PySREmulator):
    X, y = load_X_and_y_for_test()
    run_three_main_functions(emulator, X, y)


def run_three_main_functions(emulator: PySREmulator, X, y):
    emulator.fit(X, y)
    emulator.predict(X)
    emulator.predict(X, index=0)

if __name__ == '__main__':
    X, y = load_X_and_y_for_test(nb_features=2)
    print(X.shape, y.shape)
