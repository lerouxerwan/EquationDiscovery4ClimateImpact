from data.utils_dataset.npp_season_v1 import get_dataset
from optimization.optimization_bayesian.optimization_bayesian_zoo import OptimizationBayesian_2, \
    OptimizationBayesian_10, OptimizationBayesian_5, OptimizationBayesian_500
from projects.paper.utils_paper import get_opt, validation_splits
from utils.utils_log import log_info


def main(validation_size_index: int, validation_split_index: int):
    validation_size = [0.2, 0.25, 0.3][validation_size_index]
    validation_split = validation_splits[validation_split_index]
    dataset = get_dataset(validation_size=validation_size, validation_split=validation_split)
    opt = get_opt(opt_type=OptimizationBayesian_500)
    top_emulator = opt.get_top_emulator(dataset.X_train, dataset.y_train, dataset.validation_mask, dataset.X_variable_names, dataset.X_units, dataset.y_units)
    print(top_emulator.selected_equation)



if __name__ == '__main__':
    i = 0
    for j in [0, 1, 2][-1:]:
        log_info(f'i={i}, j={j}')
        main(i,j)


