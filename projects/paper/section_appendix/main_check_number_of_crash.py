from itertools import product

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from projects.paper.utils_paper import get_opt


def main_check_number_of_crashes(fast: bool, show: bool):
    validation_sizes = [0.2, 0.25, 0.3]
    validation_splits = [ValidationSplit.RANDOM, ValidationSplit.QUANTILE_WITH_BINNING, ValidationSplit.MIDDLE]
    if fast:
        validation_sizes, validation_splits = validation_sizes[:2], validation_splits[:1]
    datasets = [get_dataset(validation_size=validation_size, validation_split=validation_split)
        for validation_split, validation_size in product(validation_splits, validation_sizes)]

    opt = get_opt()
    list_percentage_of_crash = []
    for dataset in datasets:
        top_emulator, _ = opt.run(dataset.X_train, dataset.y_train, dataset.validation_mask,
                                  dataset.X_variable_names, dataset.X_units, dataset.y_units)
        ind_crash = top_emulator.run_.ind_crashes
        percentage_of_crash = float(100 * ind_crash.sum() / len(ind_crash))
        list_percentage_of_crash.append(percentage_of_crash)
    # Print everything
    print(f'Percentages of crashes for {type(opt).__name__} = {list_percentage_of_crash}')


if __name__ == '__main__':
    main_check_number_of_crashes(False, True)