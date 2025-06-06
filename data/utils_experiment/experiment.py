import json
import os
import os.path as op
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from typing import Optional, Any

import pandas as pd
from pysr import TensorBoardLoggerSpec
from sympy import Expr

from data.utils_experiment.utils_experiment import string_to_list_int
from data.utils_experiment.utils_experiment_path import CSV_FILENAME, \
    JSON_FILENAME, CHILDREN_FILENAME, PARENT_FILENAME
from emulator.utils_hyperparameter_search.utils_column_names import PARAMS_EMULATOR_COLUMN_NAME, \
    SELECTED_FEATURE_INDEXES_COLUMN_NAME, RMSE_VALIDATION_COLUMN_NAME
from utils.utils_json_loader import string_to_dict
from utils.utils_log import log_info


@dataclass
class Experiment(object):
    """Object to handle results/plots from runs (df_cv_results, non default params, tensorboard logs)"""
    experiment_path: str

    def __post_init__(self):
        #  Create folder if needed
        if not op.exists(self.experiment_path):
            os.makedirs(self.experiment_path)

    """Search cv results"""

    @property
    def filepath_search_result(self) -> str:
        return op.join(self.experiment_path, CSV_FILENAME)

    @property
    def filepath_non_default_params(self) -> str:
        return op.join(self.experiment_path, JSON_FILENAME)

    @property
    def filepath_children(self) -> str:
        return op.join(self.experiment_path, CHILDREN_FILENAME)

    @property
    def filepath_parent(self) -> str:
        return op.join(self.experiment_path, PARENT_FILENAME)

    @property
    def filepath_duration(self) -> str:
        return op.join(self.experiment_path, 'duration.txt')

    @property
    def filepath_symbolic_link(self) -> str:
        return op.join(self.experiment_path, 'plot_path')


    @property
    def df_cv_results(self) -> pd.DataFrame:
        df_cv_results = pd.read_csv(self.filepath_search_result, index_col=0)
        df_cv_results[PARAMS_EMULATOR_COLUMN_NAME] = df_cv_results[PARAMS_EMULATOR_COLUMN_NAME].apply(string_to_dict)
        df_cv_results[SELECTED_FEATURE_INDEXES_COLUMN_NAME] = df_cv_results[SELECTED_FEATURE_INDEXES_COLUMN_NAME].apply(string_to_list_int)
        return df_cv_results

    @cached_property
    def best_series(self) -> pd.Series:
        """Series that corresponds to the set of hyperparameters with the best results"""
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

    def print_search_results(self):
        log_info(f'Best params/results from the hyperparameter search:\n')
        s = (f' RMSE Validation={round(self.best_rmse_validation, 3)} with equation of complexity {self.best_complexity}: {self.best_expr}\n '
                f'using the hyperparameters: {self.best_params}\n'
                f'experiment_path: {self.experiment_path}')
        log_info(s)

    """Tensorboard Logging"""

    @property
    def filepath_tensorboard_command(self) -> str:
        return op.join(self.experiment_path, 'tensorboard_command.txt')

    @property
    def log_dir(self) -> str:
        return op.join(self.experiment_path, 'logs')

    def get_logger_spec(self, log_interval: int = 1) -> Optional[TensorBoardLoggerSpec]:
        """Create a logger only if the log has not yet been saved"""
        log_already_saved = op.exists(self.log_dir) and (len(os.listdir(self.log_dir)) == 1)
        return None if log_already_saved else TensorBoardLoggerSpec(log_dir=self.log_dir, log_interval=log_interval)

    """Remove folder"""

    def remove_folder(self):
        # Remove files
        filepaths = [self.filepath_non_default_params, self.filepath_search_result,
                     self.filepath_children, self.filepath_parent, self.filepath_duration,
                     self.filepath_tensorboard_command]
        if op.exists(self.log_dir):
            filepaths += [op.join(self.log_dir, f) for f in os.listdir(self.log_dir)]
        for filepath in filepaths:
            if op.exists(filepath):
                os.remove(filepath)
        # Remove folders log_dir and experiment_path
        for folder in [self.log_dir, self.experiment_path]:
            if op.exists(folder):
                os.rmdir(folder)
        # Remove even the dataset folder, if it is empty
        dataset_dir = op.dirname(self.experiment_path)
        if op.exists(dataset_dir) and (not os.listdir(dataset_dir)):
            os.rmdir(dataset_dir)







