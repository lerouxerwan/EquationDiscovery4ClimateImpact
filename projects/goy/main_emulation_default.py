from data.utils_dataset.dataset import Dataset
from data.utils_dataset.npp_season_v1 import get_dataset
from emulator.emulator import Emulator
from plot.plot_diagnosis import plot_diagnosis
from projects.goy.utils_goy import get_goy_dataset_without_validation
from utils.utils_log import log_info


def main_emulation_default(nb_variables: int):
    dataset = get_goy_dataset_without_validation(nb_variables)
    # Par default le batchsize est de 50, mais je test aussi 100 (et niterations =100 ou 200 ou 1000)
    emulator = Emulator(batching=True, niterations=200, batch_size=50)
    emulator.fit(dataset.X_train, dataset.y_train, variable_names=dataset.X_variable_names)
    plot_diagnosis(emulator, dataset, show=False)


if __name__ == '__main__':
    main_emulation_default(nb_variables=10)
