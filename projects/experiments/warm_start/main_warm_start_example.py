from data.utils_dataset.npp_season_v1 import get_dataset
from emulator.emulator import Emulator
from plot.plot_diagnosis import plot_diagnosis
from plot.workflow import fit

if __name__ == '__main__':
    dataset= get_dataset()

    # Old emulator
    params_emulator = {'optimize_probability': 0.1, 'population_size': 25, 'populations': 86,
                       'weight_mutate_constant': 0.22598525323963287, 'weight_optimize': 0.0001,
                       'niterations': 10, 'ncycles_per_iteration': 5000, 'turbo': True}
    emulator = Emulator(**params_emulator)
    fit(emulator, dataset)

    # New emulator
    # emulator.warm_start = True
    # emulator.weight_optimize = 100
    # emulator.populations = 100
    # fit(emulator, dataset)

    # Plot the new emulator
    plot_diagnosis(emulator, dataset)