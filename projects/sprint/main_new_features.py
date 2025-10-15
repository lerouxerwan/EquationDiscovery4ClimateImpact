from data.utils_dataset.npp_season_v1 import get_dataset
from emulator.emulator import Emulator
from emulator.utils_emulator import Config
from plot.plot_diagnosis import plot_diagnosis

def main_gaussian():
    # Config.automatic_loading_and_saving = False
    dataset = get_dataset()
    emulator = Emulator(niterations=1, gaussian_fit=True,
                        X_variable_names_for_gaussian_fit=dataset.X_variable_names,
                        y_variable_name_for_gaussian_fit=dataset.y_variable_names[0])
    emulator.fit(dataset.X_train, dataset.y_train, dataset.validation_mask,
                 X_units=dataset.X_units, y_units=dataset.y_units, variable_names=dataset.X_variable_names)
    # plot_loss_vs_complexity(emulator, dataset, show=True)
    plot_diagnosis(emulator, dataset, show=False)

def main_normal():
    Config.automatic_loading_and_saving = False
    emulator = Emulator()
    dataset = get_dataset()
    emulator.fit(dataset.X_train, dataset.y_train, dataset.validation_mask,
                 X_units=dataset.X_units, y_units=dataset.y_units, variable_names=dataset.X_variable_names)
    # plot_loss_vs_complexity(emulator, dataset, show=True)
    plot_diagnosis(emulator, dataset, show=False)