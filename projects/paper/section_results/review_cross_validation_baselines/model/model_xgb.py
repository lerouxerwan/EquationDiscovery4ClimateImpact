from typing import Any

from xgboost import XGBRegressor

from projects.paper.section_results.review_cross_validation_baselines.model.model import Model


class ModelXGB(Model):

    @property
    def estimator_type(self) -> type:
        return XGBRegressor

    @property
    def param_grid(self) -> dict[str, list[Any]]:
        return {
            'n_estimators': [10, 1000],
            "learning_rate" :[1e-2, 1],
            "max_depth": [4, 8],
        }