from typing import Any

import numpy as np
import pandas as pd

NB_THRESHOLDS = 20


def get_threshold_values() -> list[float]:
    all_thresholds = 1 + (np.geomspace(1, 1000, num=20 - 1) / 2000)
    all_thresholds = [float(f) for f in all_thresholds] + [2.]
    return all_thresholds

def get_param_grid_with_thresholds(search_cv):
    #  Build param_grid_list from param_list
    params_list = pd.DataFrame(search_cv.cv_results_)['params'].to_list()
    # For each param we build a param_grid where all hyperparameters have a fixed value,
    # except 'threshold_for_best_model_selection' that takes many values
    param_grid_list = []
    for param in params_list:
        param_grid = {param_name: [param_value] for param_name, param_value in param.items()}
        param_grid['threshold_for_best_model_selection'] = get_threshold_values()
        param_grid_list.append(param_grid)
    return param_grid_list
