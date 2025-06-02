import os
import os.path as op

import numpy as np

from data.utils_experiment.experiment import Experiment
from data.utils_experiment.utils_experiment_path import get_dataset_dir
from utils.utils_log import log_info


def get_best_experiment(X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool]) -> Experiment:
    return get_best_experiments(X, y, validation_mask, nb_top_experiments=1)[0]

def get_best_experiments(X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool], nb_top_experiments: int) -> list[Experiment]:
    log_info('Load best search experiments...')
    dataset_dir = get_dataset_dir(X, y, validation_mask)
    experiment_paths = [op.join(dataset_dir, emulator_folder) for emulator_folder in os.listdir(dataset_dir)]
    experiments = [Experiment(experiment_path) for experiment_path in experiment_paths]
    experiments = [experiment for experiment in experiments
                          if op.exists(experiment.filepath_search_result)]
    return sorted(experiments, key=lambda x: x.best_rmse_validation)[:nb_top_experiments]


