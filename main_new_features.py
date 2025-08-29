from data.utils_dataset.npp_season_v1 import dataset_npp_season_v1
from emulator.emulator import Emulator
from plot.utils_plot import plot_diagnosis

emulator = Emulator(niterations=1)
dataset = dataset_npp_season_v1
emulator.fit(dataset.X_train, dataset.y_train, variable_names=dataset.X_variables_names)
plot_diagnosis(emulator, dataset, show=False)