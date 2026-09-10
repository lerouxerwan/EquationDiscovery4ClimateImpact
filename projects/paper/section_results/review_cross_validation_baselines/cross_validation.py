from dataclasses import dataclass
from typing import Optional

from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

from emulator.utils_hyperparameter_search.utils_params_distribution import get_param_distributions
from projects.paper.section_results.review_cross_validation_baselines.model.model import Model
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


    def fit(self, X, y):
        self.search_cv.fit(X, y)

    def error(self, X, y):
        y_predict = self.search_cv.best_estimator_.predict(X)
        return ERROR_FUNCTION(y, y_predict)