
"""
The issue is that the default "loss" column of the attribute "equations_" does not match the
mean_squared_error computed with results from the "predict" method
(and thus sometimes some pareto front are not correct under the loss with the "predict" method)
"""
import numpy as np
from pysr import PySRRegressor
from scipy.stats import norm
from sklearn.metrics import mean_squared_error

def main():
    # Load toy dataset, containing 100 datapoints with one feature and one target
    size = 100
    X = np.expand_dims(np.arange(size), axis=-1).astype(float)
    y = X[:, 0] ** 2 - 2 * X[:, 0] + 3 + norm.rvs(loc=0, scale=1, size=size, random_state=42)

    # Fit model
    model = PySRRegressor(niterations=1, verbosity=0)
    model.fit(X, y)

    # Show loss from the "equations_" dataframe
    loss_values_from_equations_dataframe = [float(value) for value in model.equations_['loss'].values]
    print('Loss from equations dataframe:', loss_values_from_equations_dataframe, '\n')

    # Show loss found using the predict method
    y_predict_list = [model.predict(X, index=index) for index in range(len(model.equations_))]
    loss_values_from_predict_method = [mean_squared_error(y, y_predict) for y_predict in y_predict_list]
    print('Loss from predict method:', loss_values_from_predict_method)

if __name__ == '__main__':
    main()