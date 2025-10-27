from pysr import PySRRegressor

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from emulator.emulator import Emulator
from plot.by_split.plot_loss_vs_complexity import plot_loss_vs_complexity
from plot.by_split.plot_scatter import plot_scatter
from plot.by_split.plot_uncertainty_vs_error import plot_uncertainty_vs_error
from plot.plot_diagnosis import plot_diagnosis


def main_gaussian():
    dataset = get_dataset(validation_split=ValidationSplit.NONE)
    emulator = Emulator(gaussian_fit=True,
                        X_variable_names_for_gaussian_fit=dataset.X_variable_names,
                        y_variable_name_for_gaussian_fit=dataset.y_variable_names[0],
                        niterations=3)
    emulator.fit(dataset.X_train, dataset.y_train, variable_names=dataset.X_variable_names)
    # plot_loss_vs_complexity(emulator, dataset, show=True)
    plot_uncertainty_vs_error(emulator, dataset, show=True)
    # plot_scatter(emulator, dataset, show=True)
    # plot_diagnosis(emulator, dataset, show=False)

# def main_normal():
#     emulator = PySRRegressor(timeout_in_seconds=5)
#     dataset = get_dataset(validation_split=ValidationSplit.NONE)
#     emulator.fit(dataset.X_train, dataset.y_train, X_units=dataset.X_units, y_units=dataset.y_units, variable_names=dataset.X_variable_names)
#     print(emulator.equations_['loss'])
#     plot_loss_vs_complexity(emulator, dataset, show=True)
#     plot_diagnosis(emulator, dataset, show=False)

if __name__ == '__main__':
    main_gaussian()
    # main_normal()