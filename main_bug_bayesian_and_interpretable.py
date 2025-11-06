from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from emulator.emulator_with_search import EmulatorWithSearch
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues, get_param_name_to_values
from plot.plot_diagnosis import plot_diagnosis

if __name__ == '__main__':
    fast = True
    dataset = get_dataset(validation_size=0.25, validation_split=ValidationSplit.QUANTILE_WITH_BINNING)
    params_emulator = {'timeout_in_seconds': 60*60, 'gaussian_fit': True,
                       'X_variable_names_for_gaussian_fit': dataset.X_variable_names,
                       'y_variable_name_for_gaussian_fit': dataset.y_variable_names[0],
                       'interpretable_mode': True}
    param_grid = get_param_name_to_values(ParamNameToValues.DEFAULT_CENTRED)
    if fast:
        param_grid.pop('niterations')
        params_emulator['niterations'] = 3
    params_search = {'param_grid': param_grid,
                     'search_style': 'random', 'n_iter': 4 if fast else 200, 'n_jobs': 1 if fast else 1}
    emulator = EmulatorWithSearch(**params_emulator, **params_search)
    emulator.fit(dataset.X_train, dataset.y_train, dataset.validation_mask, dataset.X_variable_names)
    plot_diagnosis(emulator, dataset)