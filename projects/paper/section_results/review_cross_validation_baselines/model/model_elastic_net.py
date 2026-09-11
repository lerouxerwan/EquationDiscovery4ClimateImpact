from typing import Any

from sklearn.linear_model import ElasticNet

from projects.paper.section_results.review_cross_validation_baselines.model.model_lasso import ModelLasso


class ModelElasticNet(ModelLasso):

    @property
    def estimator_type(self) -> type:
        return ElasticNet

    @property
    def param_grid(self) -> dict[str, list[Any]]:
        param_grid = super().param_grid
        param_grid['l1_ratio'] = [5e-2, 5]
        return param_grid
