import json
import os
import os.path as op
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Any

import numpy as np
import pandas as pd
from pysr import TensorBoardLoggerSpec
from sympy import Expr

from data.utils_experiment.utils_experiment_path import CSV_FILENAME, \
    JSON_FILENAME, SYMBOLIC_LINK_FILENAME
from emulator.utils_hyperparameter_search.utils_column_names import PARAMS_EMULATOR_COLUMN_NAME, \
    get_cv_results_column_name, RMSE_VALIDATION_COLUMN_NAME, COMPLEXITY_COLUMN_NAME, \
    VARIABLE_NAMES_COLUMN_NAME, RMSE_TEST_COLUMN_NAME, FIT_TIME_COLUMN_NAME, PARAMS_COLUMN_NAME, EXPR_COLUMN_NAME
from plot.utils_metric.utils_metric_function import mean_relative_absolute_error
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

    @property
    def run_id(self) -> str:
        return op.basename(self.experiment_path)

    @property
    def output_directory(self) -> str:
        return op.dirname(self.experiment_path)


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
    def expr_column_name(self) -> str:
        return get_cv_results_column_name(self.model_selection, EXPR_COLUMN_NAME)

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
    def mean_fit_time(self) -> float:
        """Mean duration for the fit (between all the hyperparameter settings tested)"""
        return self.df_cv_results[FIT_TIME_COLUMN_NAME].mean()


    @property
    def top_rmse_validation(self) -> float:
        return self.top_series.loc[self.rmse_validation_column_name]

    @property
    def top_complexity(self) -> int:
        return self.top_series.loc[self.complexity_column_name]

    @property
    def top_expr(self) -> Expr:
        return self.top_series.loc[self.expr_column_name]

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
        for column_name in [PARAMS_COLUMN_NAME, PARAMS_EMULATOR_COLUMN_NAME]:
            df_cv_results[column_name] = df_cv_results[column_name].apply(string_to_dict)
        return df_cv_results

    """Metric for some experiments"""

    @property
    def percentage_of_best_same_as_validated(self) -> int:
        return int(100 * self.ind_best_same_as_validated.mean())

    @property
    def ind_best_same_as_validated(self) -> pd.Series:
        data = [c1 == c2 for c1, c2 in zip(self.get_complexity_values("best"), self.get_complexity_values("validated"))]
        return pd.Series(index=self.df_cv_results.index, data=data)

    def get_complexity_values(self, model_selection: str) -> np.ndarray:
        return self.df_cv_results[get_cv_results_column_name(model_selection, COMPLEXITY_COLUMN_NAME)].values

    @property
    def mean_difference_in_rmse_validation_for_best_not_same_as_validated(self) -> float:
        rmse_validation_best = self.get_rmse_validation_values_for_best_not_same_as_validation('best')
        rmse_validation_validated = self.get_rmse_validation_values_for_best_not_same_as_validation('validated')
        assert all([rmse_best >= rmse_validated for rmse_best, rmse_validated in zip(rmse_validation_best, rmse_validation_validated)])
        return -mean_relative_absolute_error(rmse_validation_best, rmse_validation_validated)

    def get_rmse_validation_values_for_best_not_same_as_validation(self, model_selection: str) -> np.ndarray:
        series_rmse_validation = self.df_cv_results[get_cv_results_column_name(model_selection, RMSE_VALIDATION_COLUMN_NAME)]
        return series_rmse_validation.loc[~self.ind_best_same_as_validated].values





    """Fit information"""

    def save_fit_information(self, duration: str, verbose: bool = True) -> None:
        if verbose:
            log_info(f"Experiment path={self.experiment_path}")
        #  Print and save fit information to file (for the duration & the tensorboard command)
        filepath_to_fit_information = {
            self.filepath_duration: f"duration for the fit={duration}",
            self.filepath_tensorboard_command: f"tensorboard --logdir {self.log_dir}",
        }
        for filepath, fit_information in filepath_to_fit_information.items():
            if verbose:
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

    def get_logger_spec(self, log_interval: int = 1) -> Optional[TensorBoardLoggerSpec]:
        """Create a logger only if the log has not yet been saved"""
        log_already_saved = op.exists(self.log_dir) and (len(os.listdir(self.log_dir)) == 1)
        return None if log_already_saved else TensorBoardLoggerSpec(log_dir=self.log_dir, log_interval=log_interval)


    """Filepaths"""

    @property
    def filepath_tensorboard_command(self) -> str:
        return op.join(self.experiment_path, 'tensorboard_command.txt')

    @property
    def filepath_search_result(self) -> str:
        return op.join(self.experiment_path, CSV_FILENAME)

    @property
    def filepath_non_default_params(self) -> str:
        return op.join(self.experiment_path, JSON_FILENAME)

    @property
    def filepath_symbolic_link(self) -> str:
        return op.join(self.experiment_path, SYMBOLIC_LINK_FILENAME)

    @property
    def filepath_checkpoint(self) -> str:
        return op.join(self.experiment_path, 'checkpoint.pkl')

    @property
    def filepath_hall_of_fame(self) -> str:
        return op.join(self.experiment_path, 'hall_of_fame.csv')

    @property
    def filepath_hall_of_fame_bak(self) -> str:
        return op.join(self.experiment_path, 'hall_of_fame.csv.bak')

    """Remove folder"""

    def remove_folder(self):
        # Remove files
        filepaths = [self.filepath_non_default_params, self.filepath_search_result,
                     self.filepath_duration, self.filepath_tensorboard_command,
                     self.filepath_checkpoint, self.filepath_hall_of_fame, self.filepath_hall_of_fame_bak]
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







