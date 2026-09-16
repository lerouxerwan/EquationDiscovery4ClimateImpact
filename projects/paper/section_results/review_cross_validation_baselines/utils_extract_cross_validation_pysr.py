from itertools import product

import pandas as pd

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.utils_dataset_values import load_dataset_values
from emulator.emulator import Emulator
from emulator.emulator_with_search import EmulatorWithSearch
from plot.utils_metric.metric import compute_loss
from projects.paper.section_results.review_cross_validation_baselines.main_cross_validation import \
    CROSS_VALIDATION_METRICS
from projects.paper.utils_paper import validation_splits, validation_sizes, get_opt
from utils.utils_log import log_info


def get_optimal_hyperparameters():
    datasets = [get_dataset(validation_size=validation_size, validation_split=validation_split)
                for validation_split, validation_size in product(validation_splits, validation_sizes)]
    optimization = get_opt()
    rmse_series_list = []
    for j, dataset in enumerate(datasets, 1):
        emulator_with_search = optimization.get_top_emulator(dataset.X_train, dataset.y_train, dataset.validation_mask,
                         dataset.X_variable_names, dataset.X_units, dataset.y_units)
        assert isinstance(emulator_with_search, EmulatorWithSearch)
        df = emulator_with_search.run_.df_cv_results
        rmse_series_list.append(df['mean_test_RMSE'])
    df_rmse = pd.concat(rmse_series_list, axis=1)
    mean_rmse = df_rmse.mean(axis=1).sort_values()
    index_with_minimum_mean_rmse = mean_rmse.index.values[0]
    optimal_params_emulator = df.loc[index_with_minimum_mean_rmse, 'params_emulator']
    return optimal_params_emulator


def run_cross_validation_pysr():
    # Fit on the whole train set, PySR interpretable with the optimal hyperparameters
    X_train, y_train, X_test, y_test, years_train, _, X_units, y_units, _, _, X_variable_names, y_variable_names, validation_mask, _ \
        = load_dataset_values('NPP_season_and_annual_season.csv', "RCP85", "RCP45")
    emulator = Emulator(**get_optimal_hyperparameters())
    log_info('Run fit with optimal hyperparameters')
    emulator.fit(X_train, y_train, variable_names=X_variable_names, X_units=X_units, y_units=y_units)
    # Check predictions on the test set
    scores = []
    for metric in CROSS_VALIDATION_METRICS:
        train_loss=compute_loss(y_train, emulator.predict(X_train), metric)
        test_loss=compute_loss(y_test, emulator.predict(X_test), metric)
        scores.append(f'{round(train_loss, 2)} \\& {round(test_loss, 2)}')
    print(scores)

if __name__ == '__main__':
    run_cross_validation_pysr()
    # get_optimal_hyperparameters()