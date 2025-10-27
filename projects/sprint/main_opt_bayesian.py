from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from emulator.emulator_with_search import EmulatorWithSearch
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues, get_param_name_to_values

dataset = get_dataset(validation_size=0.3, validation_split=ValidationSplit.QUANTILE_WITH_BINNING)
params_emulator = {'model_selection': 'best', 'timeout_in_seconds': 60*30, 'gaussian_fit': True,
                   'X_variable_names_for_gaussian_fit': dataset.X_variable_names,
                   'y_variable_name_for_gaussian_fit': dataset.y_variable_names[0]}
params_search = {'param_grid': get_param_name_to_values(ParamNameToValues.DEFAULT_CENTRED), 'search_style': 'random',
                 'n_iter': 500, 'n_jobs': -1}
emulator = EmulatorWithSearch(**params_emulator, **params_search)
emulator.fit(dataset.X_train, dataset.y_train, dataset.validation_mask,
             dataset.X_variable_names, dataset.X_units, dataset.y_units)