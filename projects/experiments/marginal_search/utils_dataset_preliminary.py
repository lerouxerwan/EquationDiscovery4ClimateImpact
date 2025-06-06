from copy import deepcopy

import numpy as np

from data.utils_dataset.npp_season_v1 import dataset_npp_season_v1
from data.utils_dataset.validation_split import ValidationSplit
from plot.workflow import workflow


def get_dataset_preliminary_test():
    dataset = deepcopy(dataset_npp_season_v1)
    # modify by hand the dataset
    dataset.X_train = np.concat([dataset.X_train, dataset.X_test], axis=0)
    dataset.y_train = np.concat([dataset.y_train, dataset.y_test], axis=0)
    dataset.validation_mask = np.ones(len(dataset.y_train)).astype(bool)
    dataset.validation_mask[:len(dataset.years_train)] = False
    dataset.years_train = np.concat([dataset.years_train, dataset.years_test])
    dataset.validation_split = ValidationSplit.PRELIMINARY_TEST
    return dataset

if __name__ == '__main__':
    dataset = get_dataset_preliminary_test()
    workflow(dataset, {'niterations': 2}, {'n_iter': 2})
