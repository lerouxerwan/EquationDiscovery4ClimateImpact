from data.utils_dataset.npp_season_v1 import dataset_npp_season_v1
from plot.utils_plot import plot_diagnosis
from tests.emulator.utils_tests_emulator import load_pysr_emulator_for_test


def test_plot_with_emulator():
    emulator = load_pysr_emulator_for_test()
    dataset = dataset_npp_season_v1
    emulator.fit(dataset.X_train, dataset.y_train)
    plot_diagnosis(emulator, dataset, show=None)








