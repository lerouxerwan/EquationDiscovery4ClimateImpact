import time
from collections import OrderedDict

import pandas as pd
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from plot.utils_metric.metric import Metric, compute_loss, metric_to_str
from projects.paper.section_results.subsect_3_analyze_best_equation.emulator_linear import EmulatorLinear
from utils.utils_latex import print_df_latex


def get_scores(emulator, metrics: list[Metric]):
    # Load dataset
    dataset = get_dataset(validation_split=ValidationSplit.NONE)
    variable_names = ['SSS_MAM', 'SST_MAM', 'SST_DJF', 'Shortwave_DJF']
    variable_indexes = [dataset.X_variable_names.index(variable_name) for variable_name in variable_names]
    X_train, X_test = dataset.X_train[:, variable_indexes], dataset.X_test[:, variable_indexes]
    y_train_true, y_test_true = dataset.y_train, dataset.y_test
    # Fit model
    start = time.time()
    emulator.fit(X_train, y_train_true)
    end = time.time()
    print(f'Fit duration: {round(end-start,1)}s')
    # Save model into JSON format.
    # clf.save_model("clf.json")
    # Predict with fitted model
    y_train_pred, y_test_pred = emulator.predict(X_train), emulator.predict(X_test)
    # Compute scores
    scores = []
    for metric in metrics:
        train_loss=compute_loss(y_train_true, y_train_pred, metric)
        test_loss=compute_loss(y_test_true, y_test_pred, metric)
        scores.append(f'{round(train_loss, 2)} \\& {round(test_loss, 2)}')
    return scores


def compare_emulators():
    metrics = [Metric.RMSE, Metric.MRAE, Metric.COR]
    emulators = [EmulatorLinear(), Lasso(), ElasticNet(), RandomForestRegressor(), MLPRegressor() ,XGBRegressor()]
    names = ['Linear regression', 'Lasso', "ElasticNet", "RandomForest", "MLP", "XGBoost"]
    emulator_name_to_scores = OrderedDict()
    for name, emulator in zip(names, emulators):
        print(f'Compute scores for {name}')
        emulator_name_to_scores[name] = get_scores(emulator, metrics)
    print(emulator_name_to_scores)
    index = [metric_to_str[metric] for metric in metrics]
    df = pd.DataFrame(emulator_name_to_scores, index=index).transpose()
    print(df.head())
    print_df_latex(df, index=True)


if __name__ == '__main__':
    compare_emulators()