import json
import os
import os.path as op
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Any

import pandas as pd
from pysr import TensorBoardLoggerSpec

from data.utils_run.utils_run import CSV_FILENAME, \
    JSON_FILENAME, params_that_do_not_impact_the_fit_results
from emulator.utils_hyperparameter_search.utils_column_names import PARAMS_EMULATOR_COLUMN_NAME, \
    RMSE_VALIDATION_COLUMN_NAME, PARAMS_COLUMN_NAME, RMSE_TRAIN_COLUMN_NAME
from utils.utils_json_loader import string_to_dict
from utils.utils_log import log_info


@dataclass
class Run(object):
    """Abstract object to handle results from a run"""
    output_directory: str
    run_id: str

    def __post_init__(self):
        #  Create folder run_directory if needed
        self.run_directory = op.join(self.output_directory, self.run_id)
        if not op.exists(self.run_directory):
            os.makedirs(self.run_directory)


    """Run for all emulator types: 1) save the duration of the fit 2) save tensorboard logs"""

    @property
    def filepaths_all_emulator(self) -> list[str]:
        return [self.filepath_duration, self.filepath_tensorboard_command]

    @property
    def filepath_duration(self) -> str:
        return op.join(self.run_directory, 'duration.txt')

    @property
    def filepath_tensorboard_command(self) -> str:
        return op.join(self.run_directory, 'tensorboard_command.txt')

    def save_fit(self, duration: str, verbose: bool = True) -> None:
        if verbose:
            log_info(f"Run directory={self.run_directory}")
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
    def log_dir(self) -> str:
        return op.join(self.run_directory, 'logs')

    @property
    def tensorboard_command(self) -> str:
        return f"tensorboard --logdir {self.log_dir}"

    def get_logger_spec(self, log_interval: int = 1) -> Optional[TensorBoardLoggerSpec]:
        """Create a logger only if the log has not yet been saved"""
        log_already_saved = op.exists(self.log_dir) and (len(os.listdir(self.log_dir)) == 1)
        return None if log_already_saved else TensorBoardLoggerSpec(log_dir=self.log_dir, log_interval=log_interval)

    """Run Emulator: 1) save checkpoint 2) save hall of fame csv  (normal and '.bak')"""

    @property
    def filepath_checkpoint(self) -> str:
        return op.join(self.run_directory, 'checkpoint.pkl')

    @property
    def filepath_hall_of_fame(self) -> str:
        return op.join(self.run_directory, 'hall_of_fame.csv')

    @property
    def filepath_hall_of_fame_bak(self) -> str:
        return op.join(self.run_directory, 'hall_of_fame.csv.bak')

    @property
    def filepaths_emulator(self) -> list[str]:
        return [self.filepath_checkpoint, self.filepath_hall_of_fame_bak, self.filepath_hall_of_fame]

    @property
    def has_been_saved(self) -> bool:
        return all([op.exists(filepath) for filepath in self.filepaths_emulator])


    """Run EmulatorWithSearch: 1) save search results as a csv filename 2) save non default params as json file"""

    @property
    def filepath_search_result(self) -> str:
        return op.join(self.run_directory, CSV_FILENAME)

    @property
    def filepath_non_default_params(self) -> str:
        return op.join(self.run_directory, JSON_FILENAME)

    @property
    def filepaths_emulator_with_search(self) -> list[str]:
        return [self.filepath_search_result, self.filepath_non_default_params]

    @property
    def top_params_emulator(self) -> dict[str, Any]:
        """Set of hyperparameters that minimizes the performance on the validation set"""
        params = self.top_series.loc[PARAMS_EMULATOR_COLUMN_NAME]
        return {k: v for k, v in params.items() if k not in params_that_do_not_impact_the_fit_results}

    @cached_property
    def top_series(self) -> pd.Series:
        """Series that corresponds to the top set of hyperparameters minimizing performance on validation set"""
        return self.df_cv_results.iloc[0]

    @cached_property
    def df_cv_results(self) -> pd.DataFrame:
        """Dataframe with search results. it is ordered based on the validation RMSE"""
        log_info('Load search results from files')
        #  Load dataframe from csv file
        df_cv_results = pd.read_csv(self.filepath_search_result, index_col=0)
        #  Handle deprecated df_cv_results files
        if RMSE_VALIDATION_COLUMN_NAME not in df_cv_results.columns:
            assert 'RMSE_validation' in df_cv_results.columns
            old_and_new_column_names = zip(['RMSE_train', 'RMSE_validation'], [RMSE_TRAIN_COLUMN_NAME, RMSE_VALIDATION_COLUMN_NAME])
            for old_column_name, new_column_name in old_and_new_column_names:
                df_cv_results[new_column_name] = df_cv_results[old_column_name]
        #  Sort the DataFrame by their predictive performance on the validation set
        df_cv_results = df_cv_results.sort_values(by=RMSE_VALIDATION_COLUMN_NAME)
        #  Cast some columns to their original type
        for column_name in [PARAMS_COLUMN_NAME, PARAMS_EMULATOR_COLUMN_NAME]:
            df_cv_results[column_name] = df_cv_results[column_name].apply(string_to_dict)
        return df_cv_results

    def save_search_results(self, df_cv_results: pd.DataFrame, non_default_params: dict[str, Any]) -> None:
        """Seave results from sklearn search 'df_cv_results' to files"""
        log_info('Save search results to files')
        #  Save a csv containing df_cv_results
        df_cv_results.to_csv(self.filepath_search_result)
        #  Save the associated json config file
        with open(self.filepath_non_default_params, 'w') as fp:
            json.dump(non_default_params, fp, sort_keys=True, indent=4)

    """Remove filepaths and folders"""

    def remove_folder(self):
        # Remove files
        filepaths = self.filepaths_all_emulator + self.filepaths_emulator + self.filepaths_emulator_with_search
        if op.exists(self.log_dir):
            filepaths += [op.join(self.log_dir, f) for f in os.listdir(self.log_dir)]
        for filepath in filepaths:
            if op.exists(filepath):
                os.remove(filepath)
        # Remove folders log_dir and run_directory
        for folder in [self.log_dir, self.run_directory]:
            if op.exists(folder):
                os.rmdir(folder)
        # Remove even the dataset folder, if it is empty
        dataset_dir = op.dirname(self.run_directory)
        if op.exists(dataset_dir) and (not os.listdir(dataset_dir)):
            os.rmdir(dataset_dir)







