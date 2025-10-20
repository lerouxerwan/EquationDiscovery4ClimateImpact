import numpy as np

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from emulator.emulator import Emulator
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

def main_normal(niterations: int, scaling_data_augmentation: int, percent_of_std: float):
    print('Normal')
    dataset = get_dataset(validation_split=ValidationSplit.NONE)
    emulator = Emulator(niterations=niterations, maxsize=20)
    if scaling_data_augmentation is None:
        X_fit, y_fit = dataset.X_train, dataset.y_train
    else:
        X_fit, y_fit = augment_target(dataset.X_train, dataset.y_train, scaling_data_augmentation, percent_of_std)
    emulator.fit(X_fit, y_fit,
                 X_units=dataset.X_units, y_units=dataset.y_units, variable_names=dataset.X_variable_names)
    # plot_loss_vs_complexity(emulator, dataset, show=True)
    plot_diagnosis(emulator, dataset, show=False)

def augment_target(X, y, scaling: int, percent_of_std: float):
    X = np.concat([X] * scaling, axis=0)
    noise_level = percent_of_std * np.std(y) / 100
    y = np.concat([y] + [np.random.normal(y, noise_level) for _ in range(scaling - 1)], axis=0)
    assert len(X) == len(y)
    return X, y

if __name__ == '__main__':
    niterations = 2000
    # main_gaussian(niterations=niterations)
    scaling = [2, 4, 6, 10][2]
    percent_of_std = [0.1, 0.5, 1][2]
    print(niterations, scaling, percent_of_std)
    main_normal(niterations=niterations, scaling_data_augmentation=scaling, percent_of_std=percent_of_std)
    print(niterations, scaling, percent_of_std)
