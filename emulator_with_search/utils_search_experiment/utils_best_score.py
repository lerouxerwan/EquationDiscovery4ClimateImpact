import os
import os.path as op

import numpy as np

from emulator_with_search.utils_search_experiment.search_experiment import SearchExperiment
from emulator_with_search.utils_search_experiment.utils_search_path import get_dataset_dir


def get_best_search_experiment(X: np.ndarray, y: np.ndarray, ind_validation: np.ndarray[bool]) -> SearchExperiment:
    return get_best_search_experiments(X, y, ind_validation, nb_top_experiments=1)[0]

def get_best_search_experiments(X: np.ndarray, y: np.ndarray, ind_validation: np.ndarray[bool], nb_top_experiments: int) -> list[SearchExperiment]:
    dataset_dir = get_dataset_dir(X, y, ind_validation)
    search_paths = [op.join(dataset_dir, emulator_folder) for emulator_folder in os.listdir(dataset_dir)]
    search_experiments = [SearchExperiment(search_path) for search_path in search_paths]
    return sorted(search_experiments, key=lambda x: x.best_rmse_validation)[:nb_top_experiments]


