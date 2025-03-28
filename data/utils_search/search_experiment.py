import json
import os
import os.path as op
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from typing import Optional, Any

import numpy as np
import pandas as pd
from pysr import TensorBoardLoggerSpec
from sympy import Expr

from data.utils_search.utils_search_path import CSV_FILENAME, \
    JSON_FILENAME, METRIC_COLUMN_NAME, CHILDREN_FILENAME, PARENT_FILENAME
from emulator_with_search.utils_cv_results.utils_df_results import RMSE_VALIDATION_COLUMN_NAME, \
    PARAMS_EMULATOR_COLUMN_NAME
from utils.utils_json_loader import string_to_dict
from utils.utils_log import log_info


@dataclass
class SearchExperiment(object):
    """Handle results from search experiments (df_cv_results, non default params, tensorboard logs)"""
    search_path: str

    def __post_init__(self):
        #  Create folder if needed
        if not op.exists(self.search_path):
            os.makedirs(self.search_path)

    """Search cv results"""

    @property
    def filepath_search_result(self) -> str:
        return op.join(self.search_path, CSV_FILENAME)

    @property
    def filepath_non_default_params(self) -> str:
        return op.join(self.search_path, JSON_FILENAME)

    @property
    def filepath_children(self) -> str:
        return op.join(self.search_path, CHILDREN_FILENAME)

    @property
    def filepath_parent(self) -> str:
        return op.join(self.search_path, PARENT_FILENAME)

    @property
    def df_cv_results(self) -> pd.DataFrame:
        df_cv_results = pd.read_csv(self.filepath_search_result, index_col=0)
        df_cv_results[PARAMS_EMULATOR_COLUMN_NAME] = df_cv_results[PARAMS_EMULATOR_COLUMN_NAME].apply(string_to_dict)
        return df_cv_results

    @cached_property
    def best_series(self) -> pd.Series:
        return self.df_cv_results.iloc[0]

    @property
    def best_params(self) -> dict[str, Any]:
        return self.best_series.loc[PARAMS_EMULATOR_COLUMN_NAME]
    
    @property
    def best_expr(self) -> Expr:
        return self.best_series.loc["selected_expr"]

    @property
    def best_complexity(self) -> int:
        return self.best_series.loc["selected_complexity"]

    @property
    def best_rmse_validation(self) -> float:
        return self.best_series.loc[RMSE_VALIDATION_COLUMN_NAME]

    def save_search_results(self, df_cv_results: pd.DataFrame, non_default_params: dict[str, Any]) -> None:
        log_info('Save search results to files')
        #  Save a csv containing df_cv_results
        df_cv_results.to_csv(self.filepath_search_result)
        #  Save the associated json config file
        with open(self.filepath_non_default_params, 'w') as fp:
            json.dump(non_default_params, fp, sort_keys=True, indent=4)

    def get_combinations_of_param_names_in_param_grid(self, nb_elements: int) -> list[tuple]:
        """Return combinations of nb_elements of param names in param_grid with float/int values"""
        param_names_in_param_grid = [param_name for param_name, param_value in self.best_params.items()
                                     if isinstance(param_value, (int, float))]
        return list(combinations(param_names_in_param_grid, nb_elements))


    def __str__(self):
        return (f' RMSE Validation={round(self.best_rmse_validation, 3)} with equation of complexity {self.best_complexity}: {self.best_expr}\n '
                f'using the hyperparameters: {self.best_params}\n'
                f'search_path: {self.search_path}')

    """Tensorboard Logging"""

    @property
    def log_dir(self) -> str:
        return op.join(self.search_path, 'logs')

    def get_logger_spec(self, log_interval: int = 1) -> Optional[TensorBoardLoggerSpec]:
        """Create a logger only if the log has not yet been saved"""
        log_already_saved = op.exists(self.log_dir) and (len(os.listdir(self.log_dir)) == 1)
        return None if log_already_saved else TensorBoardLoggerSpec(log_dir=self.log_dir, log_interval=log_interval)

    """Remove folder"""

    def remove_folder(self):
        # Remove files
        filepaths = [self.filepath_non_default_params, self.filepath_search_result,
                     self.filepath_children, self.filepath_parent]
        if op.exists(self.log_dir):
            filepaths += [op.join(self.log_dir, f) for f in os.listdir(self.log_dir)]
        for filepath in filepaths:
            if op.exists(filepath):
                os.remove(filepath)
        # Remove folders log_dir and search_path
        for folder in [self.log_dir, self.search_path]:
            if op.exists(folder):
                os.rmdir(folder)
        # Remove even the dataset folder, if it is empty
        dataset_dir = op.dirname(self.search_path)
        if op.exists(dataset_dir) and (not os.listdir(dataset_dir)):
            os.rmdir(dataset_dir)







