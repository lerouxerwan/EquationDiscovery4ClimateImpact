from data.utils_dataset.dataset import Dataset
from data.utils_dataset.npp_season_v1 import get_dataset
from emulator.emulator import Emulator
from plot.plot_diagnosis import plot_diagnosis
from projects.goy.utils_goy import get_goy_dataset_without_validation
from utils.utils_log import log_info


def main_emulation_default(nb_variables: int):
    dataset = get_goy_dataset_without_validation(nb_variables)
    emulator = Emulator(batching=True, niterations=5)
    emulator.fit(dataset.X_train, dataset.y_train, variable_names=dataset.X_variable_names)
    plot_diagnosis(emulator, dataset, show=False)


if __name__ == '__main__':
    main_emulation_default(nb_variables=10)
