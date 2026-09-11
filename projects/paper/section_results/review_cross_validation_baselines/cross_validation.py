import os
import pickle
from dataclasses import dataclass
from fileinput import filename
from functools import cached_property
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

from emulator.utils_hyperparameter_search.utils_params_distribution import get_param_distributions
from projects.paper.section_results.review_cross_validation_baselines.model.model import Model
from utils.utils_log import log_info
from utils.utils_path import DATA_PATH
from utils.utils_run import random_seed

ERROR_FUNCTION = root_mean_squared_error

@dataclass
class CrossValidator(object):
    model: Model
    cv: list[tuple]
    n_iter: int
    n_jobs: Optional[int] = None

    def __post_init__(self):
        self.search_cv = RandomizedSearchCV(
            estimator=self.model.estimator_type(),
            param_distributions=get_param_distributions(self.model.param_grid),
            cv=self.cv,
            scoring={'RMSE': make_scorer(ERROR_FUNCTION, greater_is_better=False)},
            refit='RMSE',
            return_train_score=True,
            n_jobs=self.n_jobs,
            random_state=random_seed,
            n_iter=self.n_iter,
        )
        self.best_estimator = None


    def fit(self, X, y):
        """Fit best estimator with the data"""
        filename = f'{self.model.estimator_type.__name__}_{len(self.cv)}cv_{X.shape[1]}features.pkl'
        filepath = Path(DATA_PATH) / 'cross_validation' / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if filepath.exists():
            log_info('Load pickle')
            self.best_estimator = pickle.load(open(filepath, 'rb'))
        else:
            log_info('Fit search_cv')
            self.search_cv.fit(X, y)
            self.best_estimator = self.search_cv.best_estimator_
            log_info('Save pickle')
            pickle.dump(self.best_estimator, open(filepath, "wb"))
        assert self.best_estimator is not None

    def error(self, X, y):
        """Predict with best estimator"""
        y_predict = self.best_estimator.predict(X)
        return ERROR_FUNCTION(y, y_predict)