import json
import os
import os.path as op
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import pandas as pd
from pysr import TensorBoardLoggerSpec

from search_dir.utils_search_dir import get_feature_dir, get_search_folder, CSV_FILENAME, JSON_FILENAME, get_best_params
from utils.utils_json_loader import string_to_dict
from utils.utils_log import log_info


@dataclass
class SearchDir(object):
    """Directory to save search results (df_cv_results_ranked_, non default params, tensorboard logs)"""
    search_dir: str

    def __post_init__(self):
        #  Create folder if needed
        if not op.exists(self.search_dir):
            os.makedirs(self.search_dir)

    @classmethod
    def from_search_arguments(cls, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame, validation_size: float,
                              feature_selection_name: str, select_k_features: Optional[int],
                              search_cv_type: type, n_iter: int, non_default_params: dict):
        feature_dir = get_feature_dir(X, y, validation_size, feature_selection_name, select_k_features)
        search_folder = get_search_folder(search_cv_type, n_iter, non_default_params)
        return cls(op.join(feature_dir, search_folder))

    """Search cv results"""

    @property
    def filepath_search_result(self) -> str:
        return op.join(self.search_dir, CSV_FILENAME)

    @property
    def filepath_non_default_params(self) -> str:
        return op.join(self.search_dir, JSON_FILENAME)


    @property
    def df_cv_results_ranked(self) -> pd.DataFrame:
        log_info(f'Load search results from file: {self.filepath_search_result}')
        df_cv_results_ranked = pd.read_csv(self.filepath_search_result, index_col=0)
        df_cv_results_ranked['params'] = df_cv_results_ranked['params'].apply(string_to_dict)
        return df_cv_results_ranked

    @property
    def best_params(self) -> dict[str, Any]:
        return self.df_cv_results_ranked.iloc[0].loc['params']

    def save_search_results(self, df_cv_results_ranked, non_default_params: dict[str, Any]):
        log_info('Save search results to files')
        #  Save a csv containing df_cv_results_ranked
        df_cv_results_ranked.to_csv(self.filepath_search_result)
        #  Save the associated json config file
        with open(self.filepath_non_default_params, 'w') as fp:
            json.dump(non_default_params, fp, sort_keys=True, indent=4)


    """Tensorboard Logging"""

    @property
    def log_dir(self) -> str:
        return op.join(self.search_dir, 'logs')

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
        for folder in [self.log_dir, self.search_dir]:
            if op.exists(folder):
                os.rmdir(folder)






