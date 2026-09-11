from collections import OrderedDict
from multiprocessing import cpu_count

import pandas as pd

from data.utils_dataset.utils_dataset_values import load_dataset_values
from plot.utils_metric.metric import Metric, metric_to_str
from projects.paper.section_results.review_cross_validation_baselines.cross_validation import CrossValidator
from projects.paper.section_results.review_cross_validation_baselines.model.model import Model
from projects.paper.section_results.review_cross_validation_baselines.model.model_random_forest import ModelRandomForest
from projects.paper.section_results.review_cross_validation_baselines.model.model_svm import ModelSVM
from projects.paper.section_results.review_cross_validation_baselines.utils_cross_validation import get_cv
from utils.utils_latex import print_df_latex


def get_scores(model: Model, metrics: list[Metric], fast: bool):
    niter = 2 if fast else 500
    n_jobs = 1 if fast else cpu_count() - 1
    # Load train/test datasets, and cross_validation setting (cv)
    rcp_name_train = "RCP85"
    X_train, y_train, X_test, y_test, years_train, _, X_units, y_units, _, _, X_variable_names, y_variable_names, _, _ \
        = load_dataset_values('NPP_season_and_annual_season.csv', rcp_name_train, "RCP45")
    cv = get_cv(y_train, list(years_train), rcp_name_train, fast)
    # Fit best estimator
    cross_validator = CrossValidator(model, cv, niter, n_jobs)
    cross_validator.fit_estimator(X_train, y_train)
    # Compute scores
    scores = []
    for metric in metrics:
        train_loss=cross_validator.compute_loss(X_train, y_train, metric)
        test_loss=cross_validator.compute_loss(X_test, y_test, metric)
        scores.append(f'{round(train_loss, 2)} \\& {round(test_loss, 2)}')
    return scores


def main_cross_validation(fast: bool):
    metrics = [Metric.RMSE, Metric.MRAE, Metric.COR]
    units = [' (gC year^{-1})', ' (\\%)', '']
    models = [ModelSVM(), ModelRandomForest()]
    model_name_to_scores = OrderedDict()
    for model in models:
        name = model.name
        print(f'Compute scores for {name}')
        model_name_to_scores[name] = get_scores(model, metrics, fast)
    print(model_name_to_scores)
    index = [metric_to_str[metric] + unit for metric, unit in zip(metrics, units)]
    df = pd.DataFrame(model_name_to_scores, index=index).transpose()
    print(df.head())
    print_df_latex(df, index=True)

if __name__ == '__main__':
    main_cross_validation(fast=True)