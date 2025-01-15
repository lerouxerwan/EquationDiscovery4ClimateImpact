from emulator.climate_impact_emulator.climate_impact_emulator import ClimateImpactEmulator
from emulator.climate_impact_emulator_with_search.climate_impact_emulator_with_search import \
    ClimateImpactEmulatorWithSearch
from tests.emulator.utils_tests_emulator import load_X_and_y_for_1D_test


def main_example_1d():
    X, y = load_X_and_y_for_1D_test()
    # emulator = ClimateImpactEmulator()
    emulator = ClimateImpactEmulatorWithSearch(param_grid={'niterations': [10, 20]}, n_iter=2)
    emulator.fit(X, y)
    y_predict = emulator.predict(X)
    print(y_predict)

if __name__ == '__main__':
    main_example_1d()