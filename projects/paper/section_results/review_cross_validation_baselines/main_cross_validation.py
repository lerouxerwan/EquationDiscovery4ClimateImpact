from collections import OrderedDict
from multiprocessing import cpu_count

import pandas as pd

from data.utils_dataset.utils_dataset_values import load_dataset_values
from plot.utils_metric.metric import Metric, metric_to_str
from projects.paper.section_results.review_cross_validation_baselines.cross_validation import CrossValidator
from projects.paper.section_results.review_cross_validation_baselines.model.model import Model
from projects.paper.section_results.review_cross_validation_baselines.model.model_elastic_net import ModelElasticNet
from projects.paper.section_results.review_cross_validation_baselines.model.model_lasso import ModelLasso
from projects.paper.section_results.review_cross_validation_baselines.model.model_random_forest import ModelRandomForest
from projects.paper.section_results.review_cross_validation_baselines.model.model_xgb import ModelXGB
from projects.paper.section_results.review_cross_validation_baselines.utils_cross_validation import get_cv
from utils.utils_latex import print_df_latex, str_df_latex

CROSS_VALIDATION_METRICS = [Metric.RMSE, Metric.MRAE, Metric.COR]

def get_scores(model: Model, fast: bool):
    niter = 2 if fast else 500
    n_jobs = 1 if fast else cpu_count() - 1
    # Load train/test datasets, and cross_validation setting (cv)
    rcp_name_train = "RCP85"
    X_train, y_train, X_test, y_test, years_train, _, X_units, y_units, _, _, X_variable_names, y_variable_names, validation_mask, _ \
        = load_dataset_values('NPP_season_and_annual_season.csv', rcp_name_train, "RCP45")
    cv = get_cv(y_train, list(years_train), rcp_name_train, fast)
    # Fit best estimator
    cross_validator = CrossValidator(model, cv, niter, n_jobs)
    cross_validator.fit_estimator(X_train, y_train)
    # Compute scores
    scores = []
    for metric in CROSS_VALIDATION_METRICS:
        train_loss=cross_validator.compute_loss(X_train, y_train, metric)
        test_loss=cross_validator.compute_loss(X_test, y_test, metric)
        scores.append(f'{round(train_loss, 2)} \\& {round(test_loss, 2)}')
    return scores


def main_cross_validation(fast: bool):
    units = [' (gC year^{-1})', ' (\\%)', '']
    models = [ModelXGB(), ModelRandomForest(), ModelElasticNet(), ModelLasso()][::-1]
    model_name_to_scores = OrderedDict()
    # Add linear regression with all features
    linear_regression_name = 'LR'
    model_name_to_scores[linear_regression_name] = ['2.08 \\& 1.95', '5.47 \\& 4.95', '0.59 \\& 0.55']
    # Add baseline models
    for model in models:
        name = model.name
        print(f'Compute scores for {name}')
        model_name_to_scores[name] = get_scores(model, fast)
    print(model_name_to_scores)
    # Add the results for PySR interpretable
    model_name_to_scores['ISR'] = ['2.08 \\& 1.95', '5.47 \\& 4.95', '0.59 \\& 0.55']
    # Add the result from the manuscript
    our_approach_name = 'Our approach (ISR + LR)'
    model_name_to_scores[our_approach_name] = ['1.68 \\& 1.67', '4.39 \\& 4.32', '0.65 \\& 0.61']
    # Create Dataframe
    index = [metric_to_str[metric] + unit for metric, unit in zip(CROSS_VALIDATION_METRICS, units)]
    df = pd.DataFrame(model_name_to_scores, index=index)
    # Print transposed df
    df = df.transpose()
    # Add cross validation column name
    cross_validation_column_name = 'Objective of cross validation:'
    df[cross_validation_column_name] = 'select a set of hyperparameters for the final model'
    df.loc[our_approach_name, cross_validation_column_name] = 'identify the form and features for the final equation'
    df.loc[linear_regression_name, cross_validation_column_name] = '-'
    print(df.head())
    s = str_df_latex(df, None, index=True)
    s = s.replace('final equation \\\\', 'final equation \\\\ \\\\')
    print(s)

if __name__ == '__main__':
    # main_cross_validation(fast=True)
    main_cross_validation(fast=False)