from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from plot.by_split.plot_scatter import plot_scatter_side_by_side
from plot.by_split.plot_scatter_and_time_series import plot_scatter_and_time_series
from plot.by_split.plot_time_series_side_by_side import plot_time_series_side_by_side
from projects.paper.utils_paper import get_opt


def main_diagnosis_best_equation(show: bool = False):
    dataset = get_dataset(validation_size=0.2, validation_split=ValidationSplit.QUANTILE_WITH_BINNING)
    opt = get_opt()
    top_emulator, _  = opt.run(dataset.X_train, dataset.y_train, dataset.validation_mask,
             dataset.X_variable_names, dataset.X_units, dataset.y_units)
    plot_scatter_side_by_side(top_emulator, dataset, show=show)
    plot_time_series_side_by_side(top_emulator, dataset, show=show)
    plot_scatter_and_time_series(top_emulator, dataset, show=show)
    # plot_climato_side_by_side(top_emulator, dataset, show=show)

if __name__ == '__main__':
    main_diagnosis_best_equation(show=False)