import json
import os
import os.path as op
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import pandas as pd
from pysr import TensorBoardLoggerSpec
from sklearn.base import BaseEstimator
from sympy import Expr

from emulator_with_search.search_experiment.utils_search_experiment import CSV_FILENAME, \
    JSON_FILENAME, get_non_default_params, METRIC_COLUMN_NAME
from utils.utils_json_loader import string_to_dict
from utils.utils_log import log_info


@dataclass
class SearchExperiment(object):
    """Handle results from search experiments (df_cv_results_ranked_, non default params, tensorboard logs)"""
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
    def df_cv_results_ranked(self) -> pd.DataFrame:
        log_info(f'Load search results from file: {self.filepath_search_result}')
        df_cv_results_ranked = pd.read_csv(self.filepath_search_result, index_col=0)
        df_cv_results_ranked['params'] = df_cv_results_ranked['params'].apply(string_to_dict)
        return df_cv_results_ranked

    @property
    def best_series(self) -> pd.Series:
        return self.df_cv_results_ranked.iloc[0]

    @property
    def best_params(self) -> dict[str, Any]:
        return self.best_series.loc['params']
    
    @property
    def best_expr(self) -> Expr:
        return self.best_series.loc["selected_expr"]

    @property
    def best_rmse_validation(self) -> float:
        return np.sqrt(-float(self.best_series.loc[METRIC_COLUMN_NAME]))

    def save_search_results(self, df_cv_results_ranked: pd.DataFrame, estimator:BaseEstimator) -> None:
        log_info('Save search results to files')
        #  Save a csv containing df_cv_results_ranked
        df_cv_results_ranked.to_csv(self.filepath_search_result)
        #  Save the associated json config file
        with open(self.filepath_non_default_params, 'w') as fp:
            json.dump(get_non_default_params(estimator), fp, sort_keys=True, indent=4)


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
        filepaths = [self.filepath_non_default_params, self.filepath_search_result]
        filepaths += [op.join(self.log_dir, f) for f in os.listdir(self.log_dir)]
        for filepath in filepaths:
            if op.exists(filepath):
                os.remove(filepath)
        # Remove folders
        for folder in [self.log_dir, self.search_path]:
            if op.exists(folder):
                os.rmdir(folder)






