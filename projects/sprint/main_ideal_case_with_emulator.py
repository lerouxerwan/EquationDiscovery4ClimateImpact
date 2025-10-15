import numpy as np

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from emulator.emulator import Emulator
from plot.plot_diagnosis import plot_diagnosis
from plot.workflow import fit
from utils.utils_run import random_seed


def get_X_y_from_normal(nb_samples: int):
    np.random.seed(random_seed)
    samples = []
    x_array = np.linspace(0, 1, num=nb_samples)
    for x in x_array:
        mu = x * x + x - 1
        sigma = 1
        sample = np.random.normal(loc=mu, scale=sigma, size=1)[0]
        samples.append(sample)
    X = np.expand_dims(x_array, axis=1)
    y = np.array(samples)
    return X, y


def fit_gaussian(X, y) -> Emulator:
    emulator = Emulator(niterations=1, gaussian_fit=True,
                        X_variable_names_for_gaussian_fit=['x'], y_variable_name_for_gaussian_fit='y')
    emulator.fit(X, y, variable_names=['x'])
    return emulator

if __name__ == '__main__':
    # X, y = get_X_y_from_normal(nb_samples=1000)
    # emulator = fit_gaussian(X, y)

    dataset = get_dataset()
    emulator = Emulator(niterations=1, gaussian_fit=True,
                        X_variable_names_for_gaussian_fit=dataset.X_variable_names,
                        y_variable_name_for_gaussian_fit=dataset.y_variable_names[0])
    emulator.fit(dataset.X_train, dataset.y_train, None,
                 dataset.X_variable_names)
    plot_diagnosis(emulator, dataset)
    print(emulator.loss_list)
    print(emulator.expr_list)