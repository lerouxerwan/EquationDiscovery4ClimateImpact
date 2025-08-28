from pysr import PySRRegressor

from utils_dataset import variable_names, X_units, y_units, X, y
from utils_model import params


def main():
    model = PySRRegressor(**params)
    model.fit(X, y, variable_names=variable_names, X_units=X_units, y_units=y_units)

    print(X.sum())
    print(y.sum())
    print(len(model.equations_))
    print(model.equations_['complexity'].to_list())


if __name__ == '__main__':
    main()