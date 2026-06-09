from emulator.emulator import Emulator
from plot.plot_diagnosis import plot_diagnosis
from projects.gaussian_lorenz.dataset_lorenz import get_dataset_lorenz


def main_emulation_default_with_gaussian_fit(state_variable_index: int, n_trajectories: int):
    dataset = get_dataset_lorenz(state_variable_index, n_trajectories)
    emulator = Emulator(gaussian_fit=True, X_variable_names_for_gaussian_fit=dataset.X_variable_names,
                        y_variable_name_for_gaussian_fit='y')
    emulator.fit(dataset.X_train, dataset.y_train, variable_names=dataset.X_variable_names)
    plot_diagnosis(emulator, dataset, show=True)

if __name__ == '__main__':
    for state_variable_index in list(range(3))[2:]:
        main_emulation_default_with_gaussian_fit(state_variable_index, 4)