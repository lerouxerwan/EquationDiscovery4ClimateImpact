from emulator.emulator import Emulator
from plot.plot_diagnosis import plot_diagnosis
from projects.goy.utils_goy import get_goy_dataset


def main_emulation_default(nb_variables: int, with_validation: bool):
    dataset = get_goy_dataset(nb_variables, with_validation)
    # Par default le batchsize est de 50, mais je test aussi 100 (et niterations =100 ou 200 ou 1000)
    emulator = Emulator(batching=True, niterations=100, batch_size=50)
    emulator.fit(dataset.X_train, dataset.y_train, validation_mask=dataset.validation_mask,
                 variable_names=dataset.X_variable_names)
    plot_diagnosis(emulator, dataset, show=True)


if __name__ == '__main__':
    main_emulation_default(nb_variables=10, with_validation=False)
