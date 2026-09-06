from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from plot.by_split.plot_scatter import plot_scatter_side_by_side
from plot.by_split.plot_scatter_and_time_series import plot_scatter_and_time_series
from plot.by_split.plot_time_series_side_by_side import plot_time_series_side_by_side
from projects.paper.section_results.subsect_3_analyze_best_equation.emulator_linear import EmulatorLinear


def main_diagnosis_baseline_equation_with_four_selection_features(show: bool = False):
    dataset = get_dataset(validation_split=ValidationSplit.NONE)
    variable_names = ['SSS_MAM', 'SST_MAM', 'SST_DJF', 'Shortwave_DJF']
    variable_indexes = [dataset.X_variable_names.index(variable_name) for variable_name in variable_names]
    emulator = EmulatorLinear(variable_names=variable_names, features_indexes=variable_indexes)
    assert emulator.variable_names is not None
    assert emulator.features_indexes is not None
    emulator.fit(dataset.X_train, dataset.y_train)
    # print(emulator.selected_expr)
    # plot_scatter_side_by_side(emulator, dataset, show=show)
    # plot_time_series_side_by_side(emulator, dataset, show=show)
    plot_scatter_and_time_series(emulator, dataset, show=show)
    # plot_climato_side_by_side(top_emulator, dataset, show=show)


if __name__ == '__main__':
    main_diagnosis_baseline_equation_with_four_selection_features(show=False)