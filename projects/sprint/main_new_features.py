from data.utils_dataset.npp_season_v1 import dataset_npp_season_v1
from emulator.emulator import Emulator
from plot.by_split.plot_loss_vs_complexity import plot_loss_vs_complexity
from plot.plot_diagnosis import plot_diagnosis
from projects.sprint.custom_complexity import complexity_mapping, complexity_mapping_basic

emulator = Emulator(niterations=20, complexity_mapping=complexity_mapping)
dataset = dataset_npp_season_v1
emulator.fit(dataset.X_train, dataset.y_train, variable_names=dataset.X_variables_names)
plot_loss_vs_complexity(emulator, dataset, show=True)
# plot_diagnosis(emulator, dataset, show=False)