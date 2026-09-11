from typing import Any

from sklearn.linear_model import Lasso

from projects.paper.section_results.review_cross_validation_baselines.model.model import Model


class ModelLasso(Model):

    @property
    def estimator_type(self) -> type:
        return Lasso

    @property
    def param_grid(self) -> dict[str, list[Any]]:
        return {
            'alpha': [0.1, 10],
            'tol': [1e-5, 1e-3],
            'max_iter': [100, 10_000],
            'selection':  ['cyclic', 'random']
        }