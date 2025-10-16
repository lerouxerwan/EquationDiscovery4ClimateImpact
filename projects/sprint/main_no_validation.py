from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from emulator.emulator import Emulator
from emulator.utils_emulator import Config
from plot.plot_diagnosis import plot_diagnosis

def main_gaussian(niterations: int):
    print('Gaussian')
    dataset = get_dataset(validation_split=ValidationSplit.NONE)
    emulator = Emulator(niterations=niterations, gaussian_fit=True,
                        X_variable_names_for_gaussian_fit=dataset.X_variable_names,
                        y_variable_name_for_gaussian_fit=dataset.y_variable_names[0])
    emulator.fit(dataset.X_train, dataset.y_train, variable_names=dataset.X_variable_names)
    print(emulator.equations_)
    for i, row in emulator.equations_.iterrows():
        print(row['complexity'], row['loss'])
        print(row['equation'])
    # plot_loss_vs_complexity(emulator, dataset, show=True)
    # plot_diagnosis(emulator, dataset, show=False)

def main_normal(niterations: int):
    print('Normal')
    dataset = get_dataset(validation_split=ValidationSplit.NONE)
    emulator = Emulator(niterations=niterations)
    emulator.fit(dataset.X_train, dataset.y_train,
                 X_units=dataset.X_units, y_units=dataset.y_units, variable_names=dataset.X_variable_names)
    # plot_loss_vs_complexity(emulator, dataset, show=True)
    plot_diagnosis(emulator, dataset, show=False)

if __name__ == '__main__':
    niterations = 10000
    print(niterations)
    # main_gaussian(niterations=niterations)
    main_normal(niterations=niterations)