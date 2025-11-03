from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from emulator.emulator import Emulator
from plot.plot_diagnosis import plot_diagnosis

if __name__ == '__main__':
    dataset = get_dataset(validation_size=0.3, validation_split=ValidationSplit.QUANTILE_WITH_BINNING)
    params_emulator = {'timeout_in_seconds': 60*60, 'gaussian_fit': True,
                       'X_variable_names_for_gaussian_fit': dataset.X_variable_names,
                       'y_variable_name_for_gaussian_fit': dataset.y_variable_names[0],
                       'model_selection': 'validated', 'niterations': 2}
    emulator = Emulator(**params_emulator)
    emulator.fit(dataset.X_train, dataset.y_train, dataset.validation_mask, dataset.X_variable_names)
    plot_diagnosis(emulator, dataset)