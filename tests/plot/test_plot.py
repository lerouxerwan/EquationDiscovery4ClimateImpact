from data.utils_dataset.npp_season_v1 import dataset_npp_season_v1
from emulator.emulator import Emulator
from plot.dataset.plot_selected_features import get_selected_feature_indexes
from plot.utils_plot import plot_diagnosis


def test_plot_with_emulator():
    emulator = Emulator(niterations=1)
    dataset = dataset_npp_season_v1
    emulator.fit(dataset.X_train, dataset.y_train, variable_names=dataset.X_variables_names)
    plot_diagnosis(emulator, dataset, show=None)
    emulator.experiment_.remove_folder()

def test_selected_feature_indexes():
    selected_variable_names = ['x0', 'x10', 'x84']
    # One test with variable_names = None
    assert get_selected_feature_indexes(selected_variable_names) == [0, 10, 84]
    # One test with specified variable_names
    variables_names = [f'x{2 * i}' for i in range(50)]
    assert get_selected_feature_indexes(selected_variable_names, variables_names) == [0, 5, 42]









