from emulator.emulator import Emulator
from plot.plot_diagnosis import plot_diagnosis
from projects.goy.utils_goy import get_goy_dataset
from tests.data.utils_tests_dataset import load_X_and_y_for_test
from utils.utils_log import log_info


def main_emulation_default(nb_variables: int):
    dataset = get_goy_dataset(nb_variables, with_validation=False)
    # Par default le batchsize est de 50, mais je test aussi 100 (et niterations =100 ou 200 ou 1000)
    niterations = [50, 100, 200, 1000][-1]
    batch_size = [50, 100, 200][2]
    log_info(f"Run with {niterations} iterations and {batch_size} batches")
    emulator = Emulator(batching=True, niterations=niterations, batch_size=batch_size)
    emulator.fit(dataset.X_train, dataset.y_train, validation_mask=dataset.validation_mask,
                 variable_names=dataset.X_variable_names)
    plot_diagnosis(emulator, dataset, show=True)


def main_emulation_default_with_gaussian_fit(nb_variables: int):
    dataset = get_goy_dataset(nb_variables, with_validation=False)
    # Par default le batchsize est de 50, mais je test aussi 100 (et niterations =100 ou 200 ou 1000)
    niterations = [1, 50, 100, 200][0]
    batch_size = [50, 100, 200][0]
    log_info(f"Run with {niterations} iterations and {batch_size} batches")
    variable_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'][:]
    emulator = Emulator(niterations=niterations,
                        gaussian_fit=True, X_variable_names_for_gaussian_fit=variable_names,
                        y_variable_name_for_gaussian_fit='y', batching=True)
    # emulator = Emulator(niterations=niterations,
    #                     gaussian_fit=True, X_variable_names_for_gaussian_fit=dataset.X_variable_names,
    #                     y_variable_name_for_gaussian_fit=dataset.y_variable_names[0])
    # emulator = Emulator(batching=True, niterations=niterations, batch_size=batch_size,
    #                     gaussian_fit=True, X_variable_names_for_gaussian_fit=dataset.X_variable_names,
    #                     y_variable_name_for_gaussian_fit=dataset.y_variable_names[0])
    # train_size = 100
    # X = dataset.X_train[:train_size, :1]
    # y = dataset.y_train[:train_size]
    X = dataset.X_train
    y = dataset.y_train
    print(X)
    print(y)
    emulator.fit(X, y, variable_names=variable_names)

    plot_diagnosis(emulator, dataset, show=True)


if __name__ == '__main__':
    # main_emulation_default(nb_variables=10)
    main_emulation_default_with_gaussian_fit(nb_variables=10)
