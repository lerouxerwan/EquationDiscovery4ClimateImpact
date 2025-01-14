import numpy as np

from emulator.climate_impact_emulator.climate_impact_emulator import ClimateImpactEmulator
from emulator.climate_impact_emulator_with_search.climate_impact_emulator_with_search import ClimateImpactEmulatorWithSearch


def load_climate_impact_emulator_for_test(**kwargs) -> ClimateImpactEmulator:
    return ClimateImpactEmulator(niterations=1, **kwargs)

def load_climate_impact_emulator_with_search_for_test(**kwargs) -> ClimateImpactEmulatorWithSearch:
    return ClimateImpactEmulatorWithSearch(niterations=1, **kwargs)



def load_X_and_y_for_1D_test() -> tuple[np.ndarray, np.ndarray]:
    X = np.expand_dims(np.arange(100), axis=-1)
    y = X[:, 0] ** 2 - 2 * X[:, 0] + 3
    return X, y


def run_three_main_functions(emulator: ClimateImpactEmulator):
    X, y = load_X_and_y_for_1D_test()
    emulator.fit(X, y)
    emulator.predict(X)
    emulator.predict(X, index=0)
