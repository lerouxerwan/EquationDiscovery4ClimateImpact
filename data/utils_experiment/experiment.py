import json
import os
import os.path as op
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Any

import pandas as pd
from pysr import TensorBoardLoggerSpec

from data.utils_experiment.utils_experiment_path import CSV_FILENAME, \
    JSON_FILENAME, SYMBOLIC_LINK_FILENAME
from emulator.utils_hyperparameter_search.utils_column_names import PARAMS_EMULATOR_COLUMN_NAME, \
    get_cv_results_column_name, RMSE_VALIDATION_COLUMN_NAME, COMPLEXITY_COLUMN_NAME, \
    VARIABLE_NAMES_COLUMN_NAME, RMSE_TEST_COLUMN_NAME, FIT_TIME_COLUMN_NAME
from utils.utils_json_loader import string_to_dict
from utils.utils_log import log_info


@dataclass
class Experiment(object):
    """Object to handle results/plots from runs (df_cv_results, non default params, tensorboard logs)"""
    experiment_path: str
    model_selection: str

    def __post_init__(self):
        #  Create folder if needed
        if not op.exists(self.experiment_path):
            os.makedirs(self.experiment_path)

    """Properties depending on self.model_selection"""

    @property
    def rmse_test_column_name(self) -> str:
        return get_cv_results_column_name(self.model_selection, RMSE_TEST_COLUMN_NAME)

    @property
    def rmse_validation_column_name(self) -> str:
        return get_cv_results_column_name(self.model_selection, RMSE_VALIDATION_COLUMN_NAME)

    @property
    def complexity_column_name(self) -> str:
        return get_cv_results_column_name(self.model_selection, COMPLEXITY_COLUMN_NAME)

    @property
    def variable_names_column_name(self) -> str:
        return get_cv_results_column_name(self.model_selection, VARIABLE_NAMES_COLUMN_NAME)

    """ Top search results"""

    @cached_property
    def top_series(self) -> pd.Series:
        """Series that corresponds to the top set of hyperparameters minimizing performance on validation set"""
        return self.df_cv_results.iloc[0]

    @property
    def top_params(self) -> dict[str, Any]:
        """Set of hyperparameters that minimizes the performance on the validation set"""
        return self.top_series.loc[PARAMS_EMULATOR_COLUMN_NAME]

    @property
    def top_fit_time(self) -> float:
        """Duration for the fit"""
        return self.top_series.loc[FIT_TIME_COLUMN_NAME]

    @property
    def max_fit_time(self) -> float:
        """Max duration for the fit (between all the hyperparameter settings tested)"""
        return self.df_cv_results[FIT_TIME_COLUMN_NAME].max()


    @property
    def top_rmse_validation(self) -> float:
        return self.top_series.loc[self.rmse_validation_column_name]

    @property
    def top_complexity(self) -> int:
        return self.top_series.loc[self.complexity_column_name]

    @property
    def top_rmse_test(self) -> float:
        return self.top_series.loc[self.rmse_test_column_name]


    """Save & Load search results"""

    def save_search_results(self, df_cv_results: pd.DataFrame, non_default_params: dict[str, Any]) -> None:
        log_info('Save search results to files')
        #  Save a csv containing df_cv_results
        df_cv_results.to_csv(self.filepath_search_result)
        #  Save the associated json config file
        with open(self.filepath_non_default_params, 'w') as fp:
            json.dump(non_default_params, fp, sort_keys=True, indent=4)

    @cached_property
    def df_cv_results(self) -> pd.DataFrame:
        """Dataframe with search results. During loading, it is ordered based on the self.model_selection attribute"""
        log_info('Load search results from files')
        # Load dataframe from csv file
        df_cv_results = pd.read_csv(self.filepath_search_result, index_col=0)
        # Sort the DataFrame by their predictive performance on the validation set for self.model_selection
        df_cv_results = df_cv_results.sort_values(by=self.rmse_validation_column_name)
        # Cast some columns to their original type
        df_cv_results[PARAMS_EMULATOR_COLUMN_NAME] = df_cv_results[PARAMS_EMULATOR_COLUMN_NAME].apply(string_to_dict)
        return df_cv_results

    """Filepaths"""

    @property
    def filepath_search_result(self) -> str:
        return op.join(self.experiment_path, CSV_FILENAME)

    @property
    def filepath_non_default_params(self) -> str:
        return op.join(self.experiment_path, JSON_FILENAME)

    @property
    def filepath_symbolic_link(self) -> str:
        return op.join(self.experiment_path, SYMBOLIC_LINK_FILENAME)

    """Fit information"""

    def print_and_save_fit_information(self, duration: str) -> None:
        log_info(f"Experiment path={self.experiment_path}")
        #  Print and save fit information to file (for the duration & the tensorboard command)
        filepath_to_fit_information = {
            self.filepath_duration: f"duration for the fit={duration}",
            self.filepath_tensorboard_command: f"tensorboard --logdir {self.log_dir}",
        }
        for filepath, fit_information in filepath_to_fit_information.items():
            log_info(fit_information)
            with open(filepath, 'w') as f:
                f.write(fit_information)

    @property
    def filepath_duration(self) -> str:
        return op.join(self.experiment_path, 'duration.txt')

    @property
    def log_dir(self) -> str:
        return op.join(self.experiment_path, 'logs')

    @property
    def tensorboard_command(self) -> str:
        return f"tensorboard --logdir {self.log_dir}"

    @property
    def filepath_tensorboard_command(self) -> str:
        return op.join(self.experiment_path, 'tensorboard_command.txt')

    def get_logger_spec(self, log_interval: int = 1) -> Optional[TensorBoardLoggerSpec]:
        """Create a logger only if the log has not yet been saved"""
        log_already_saved = op.exists(self.log_dir) and (len(os.listdir(self.log_dir)) == 1)
        return None if log_already_saved else TensorBoardLoggerSpec(log_dir=self.log_dir, log_interval=log_interval)

    """Remove folder"""

    def remove_folder(self):
        # Remove files
        filepaths = [self.filepath_non_default_params, self.filepath_search_result,
                     self.filepath_duration, self.filepath_tensorboard_command]
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







