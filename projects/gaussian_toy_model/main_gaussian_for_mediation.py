from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from emulator.emulator import Emulator
from plot.plot_diagnosis import plot_diagnosis

def main_gaussian_for_mediation(show: bool = False):
    dataset = get_dataset(validation_split=ValidationSplit.NONE)
    params_emulator = {'gaussian_fit': True,
                       'X_variable_names_for_gaussian_fit': dataset.X_variable_names,
                       'y_variable_name_for_gaussian_fit': dataset.y_variable_names[0],
                       'niterations': 100}
    emulator = Emulator(**params_emulator)
    emulator.fit(dataset.X_train, dataset.y_train, dataset.validation_mask, dataset.X_variable_names)
    plot_diagnosis(emulator, dataset, show=show)


if __name__ == '__main__':
    main_gaussian_for_mediation(show=False)
