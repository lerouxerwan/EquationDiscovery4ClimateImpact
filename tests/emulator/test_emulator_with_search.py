from data.utils_dataset.npp_season_v1 import dataset_npp_season_v1
from emulator.emulator_with_search import EmulatorWithSearch


def test_emulator_validation_with_search():
    dataset = dataset_npp_season_v1
    emulator = EmulatorWithSearch(niterations=5, n_iter=3, search_style='random', scaling_factor=2,
                                  model_selection='validated',
                                  param_list_to_optimize=['adaptive_parsimony_scaling'])
    emulator.fit(dataset.X_train, dataset.y_train, validation_mask=dataset.validation_mask,
                 variable_names=dataset.X_variables_names, X_units=dataset.X_units, y_units=dataset.y_units)
    assert emulator.selected_complexity == 27
    emulator.experiment_.remove_folder()