from typing import Callable, Optional

import numpy as np
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.api import OLS, add_constant
from sympy import Expr, symbols, Symbol

from plot.by_split.utils_equation_str import get_equation_from_expr, postprocessing_for_equation


class EmulatorLinearWithUncertainty(object):
    def __init__(self, variable_names: Optional[list[str]]=None, features_indexes: Optional[list[int]]=None):
        self.variable_names = variable_names
        self.features_indexes = features_indexes
        self.fitted = None

    def get_features(self, X):
        return add_constant(X[:, self.features_indexes])

    def fit(self, X, y):
        model = OLS(y, self.get_features(X), hasconst=True)
        self.fitted = model.fit()

    def predict(self, X):
        return self.fitted.predict(self.get_features(X))

    def predict_uncertainty_interval_without_gaussian_fit(self, X):
        """By default, we consider the 95% confidence interval, i.e. there is a 95 percent probability that the real
        value of y in the population for a given value of x lies within the prediction interval. """
        y_predict = self.predict(X)
        sdev, lower, upper = wls_prediction_std(self.fitted, exog=self.get_features(X), alpha=0.05)
        lower, upper = lower - y_predict, upper - y_predict
        return np.concatenate([np.expand_dims(lower, 1), np.expand_dims(upper, 1)], axis=1)

    @property
    def intercept(self) -> float:
        return self.fitted.params[0]

    @property
    def coefficients(self) -> np.ndarray:
        return self.fitted.params[1:]

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
