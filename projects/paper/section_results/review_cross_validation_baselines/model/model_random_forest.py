from typing import Any

from sklearn.ensemble import RandomForestRegressor

from projects.paper.section_results.review_cross_validation_baselines.model.model import Model


class ModelRandomForest(Model):

    @property
    def estimator_type(self) -> type:
        return RandomForestRegressor

    @property
    def param_grid(self) -> dict[str, list[Any]]:
        return {
            'n_estimators': [10, 1000],
            "criterion" :["squared_error", "absolute_error", "friedman_mse", "poisson"],
            "max_features": ["sqrt", "log2", None],
        }