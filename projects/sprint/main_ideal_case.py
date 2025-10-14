import numpy as np
from pysr import TemplateExpressionSpec, PySRRegressor


def get_X_y_from_normal(nb_samples: int):
    samples = []
    x_array = np.linspace(0, 1, num=nb_samples)
    for x in x_array:
        mu = x * x + x - 1
        sigma = 1
        sample = np.random.normal(loc=mu, scale=sigma, size=1)[0]
        samples.append(sample)
    X = np.expand_dims(x_array, axis=1)
    y = np.array(samples)
    return X, y


def fit_gaussian(X, y) -> PySRRegressor:
    X = np.concat([X,  np.expand_dims(y, axis=1)], axis=1)
    y = np.zeros(len(X))
    variable_names = ["x", "y"]
    template = TemplateExpressionSpec(
        expressions=["f", "g"],
        variable_names=variable_names,
        combine="""
            mu = f(x)
            sigma = g(x)

            log(sigma) + (y - mu)^2 / (2 * sigma^2)
        """
    )
    emulator = PySRRegressor(niterations=10, expression_spec=template,
                             elementwise_loss="my_custom_loss(predicted, target) = predicted")
    emulator.fit(X, y, variable_names=variable_names)
    return emulator



if __name__ == '__main__':
    X, y = get_X_y_from_normal(nb_samples=1000)
    emulator = fit_gaussian(X, y)