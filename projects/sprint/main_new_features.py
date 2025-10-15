from data.utils_dataset.dataset import Dataset
from data.utils_dataset.npp_season_v1 import dataset_npp_season_v1
from data.utils_dataset.validation_split import ValidationSplit
from emulator.emulator import Emulator
from emulator.utils_emulator import Config
from plot.by_rcp.plot_climato import plot_climato
from plot.by_split.plot_loss_vs_complexity import plot_loss_vs_complexity
from plot.by_split.plot_scatter import plot_scatter_side_by_side, plot_scatter
from plot.plot_diagnosis import plot_diagnosis

Config.automatic_loading_and_saving = False
emulator = Emulator()
dataset = Dataset("NPP_season_and_annual_season.csv", "RCP85", "RCP45", 0.25, ValidationSplit.QUANTILE_WITH_BINNING)
emulator.fit(dataset.X_train, dataset.y_train, dataset.validation_mask,
             X_units=dataset.X_units, y_units=dataset.y_units, variable_names=dataset.X_variable_names)
# plot_loss_vs_complexity(emulator, dataset, show=True)
plot_diagnosis(emulator, dataset, show=False)