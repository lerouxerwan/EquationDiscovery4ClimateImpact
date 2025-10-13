from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization_baseline import OptimizationBaseline
from plot.plot_diagnosis import plot_diagnosis

dataset= Dataset("NPP_season.csv", "RCP85", "RCP45", 0.2, ValidationSplit.QUANTILE_WITH_BINNING)

def main_plot_diagnosis_top_emulator():
    opt = OptimizationBaseline('best')
    emulator = opt.get_top_emulator(dataset.X_train, dataset.y_train, dataset.validation_mask,
                                    dataset.X_variable_names, dataset.X_units, dataset.y_units)
    plot_diagnosis(emulator, dataset)



if __name__ == '__main__':
    # main_compare_nested_cv()
    main_plot_diagnosis_top_emulator()
