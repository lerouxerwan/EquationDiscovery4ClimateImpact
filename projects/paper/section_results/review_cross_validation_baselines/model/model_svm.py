from typing import Any

from sklearn.svm import LinearSVR, SVR

from projects.paper.section_results.review_cross_validation_baselines.model.model import Model


class ModelSVM(Model):

    @property
    def estimator_type(self) -> type:
        return SVR

    def get_estimator(self):
        return SVR()

    @property
    def param_grid(self) -> dict[str, list[Any]]:
        return {'C': [1.0, 10.0]}