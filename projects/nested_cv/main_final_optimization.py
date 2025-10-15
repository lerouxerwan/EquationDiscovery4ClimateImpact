from data.utils_dataset.npp_season_v1 import get_dataset
from optimization.optimization_baseline import OptimizationBaseline
from plot.plot_diagnosis import plot_diagnosis


def main_plot_diagnosis_top_emulator():
    dataset = get_dataset()
    opt = OptimizationBaseline('best')
    emulator = opt.get_top_emulator(dataset.X_train, dataset.y_train, dataset.validation_mask,
                                    dataset.X_variable_names, dataset.X_units, dataset.y_units)
    plot_diagnosis(emulator, dataset)



if __name__ == '__main__':
    # main_compare_nested_cv()
    main_plot_diagnosis_top_emulator()
