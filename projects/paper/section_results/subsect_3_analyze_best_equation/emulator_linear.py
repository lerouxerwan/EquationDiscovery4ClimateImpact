from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import LinearRegression
from sympy import Expr, symbols, Symbol

from plot.by_split.utils_equation_str import get_equation_from_expr


class EmulatorLinear(LinearRegression):
    def __init__(self, *, fit_intercept=True, copy_X=True, n_jobs=None, positive=False,
                 variable_names: list[str]=None, features_indexes: list[int]=None):
        super().__init__(fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs, positive=positive)
        assert variable_names is not None
        assert features_indexes is not None
        self.variable_names = variable_names
        self.features_indexes = features_indexes

    def fit(self, X, y, sample_weight=None):
        return super().fit(X[:, self.features_indexes], y, sample_weight)

    def predict(self, X):
        return super().predict(X[:, self.features_indexes])

    @property
    def intercept(self) -> float:
        return self.intercept_

    @property
    def coefficients(self) -> np.ndarray:
        return self.coef_

    @property
    def selected_expr(self) -> Expr:
        # x_variables = symbols(' '.join([f'x{i}' for i in range(len(self.coefficients))]))
        x_variables = symbols(' '.join(self.variable_names))
        expr = self.intercept
        if isinstance(x_variables, Symbol):
            x_variables = [x_variables]
        for coefficient, x_variable in zip(self.coefficients, x_variables):
            expr += coefficient * x_variable
        return expr

    @property
    def selected_equation(self):
        return get_equation_from_expr(self.selected_expr)


    @property
    def lambda_function(self) -> Callable:
        # Apply zip because the number of coefficients might smaller than the actual number of features in x
        return lambda x: self.intercept + sum([c * v for c, v in zip(self.coefficients, x.transpose())])

    @property
    def gaussian_fit(self):
        return False
