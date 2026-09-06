from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from sklearn.linear_model import LinearRegression
from sympy import Expr, symbols, Symbol

from plot.by_split.utils_equation_str import get_equation_from_expr, postprocessing_for_equation


class EmulatorLinear(LinearRegression):
    def __init__(self, *, fit_intercept=True, copy_X=True, n_jobs=None, positive=False,
                 variable_names: Optional[list[str]]=None, features_indexes: Optional[list[int]]=None):
        super().__init__(fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs, positive=positive)
        self.variable_names = variable_names
        self.features_indexes = features_indexes

    def fit(self, X, y, sample_weight=None):
        if self.features_indexes is None:
            return super().fit(X, y, sample_weight)
        else:
            return super().fit(X[:, self.features_indexes], y, sample_weight)

    def predict(self, X):
        if self.features_indexes is None:
            return super().predict(X)
        else:
            return super().predict(X[:, self.features_indexes])

    def predict_uncertainty_interval_without_gaussian_fit(self, X):
        assert self.features_indexes is not None
        uncertainty_intervals = np.ones(shape=(X.shape[0], 2))
        uncertainty_intervals[:, 0] *= -1
        return uncertainty_intervals

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
        selected_equation = postprocessing_for_equation(get_equation_from_expr(self.selected_expr))
        assert selected_equation == '$6.0*SSS_{MAM} - 0.e-1*SST_{DJF} - 1.0*SST_{MAM} + 0.1*Shortwave_{DJF} + 106.0$'
        # 6.2398322706731*SSS_MAM - 0.0855016821575876*SST_DJF - 1.02756017634541*SST_MAM + 0.111858974953818*Shortwave_DJF + 106.367959478915
        return '$6.2*SSS_{MAM} - 0.086*SST_{DJF} - 1.03*SST_{MAM} + 0.11*Shortwave_{DJF} + 106$'

    @property
    def lambda_function(self) -> Callable:
        # Apply zip because the number of coefficients might smaller than the actual number of features in x
        return lambda x: self.intercept + sum([c * v for c, v in zip(self.coefficients, x.transpose())])

    @property
    def gaussian_fit(self):
        return False
