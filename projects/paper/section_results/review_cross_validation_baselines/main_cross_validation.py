from collections import OrderedDict
from itertools import product
from multiprocessing import cpu_count
from typing import Optional

import numpy as np

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.utils_dataset_values import load_dataset_values
from data.utils_dataset.utils_validation_split import get_validation_mask
from data.utils_dataset.validation_split import ValidationSplit
from projects.paper.section_results.review_cross_validation_baselines.cross_validation import CrossValidator
from projects.paper.section_results.review_cross_validation_baselines.model.model_svm import ModelSVM
from projects.paper.utils_paper import validation_splits, validation_sizes


def main_cross_validation(fast: bool):
    niter = 2 if fast else 500
    n_jobs = 1 if fast else cpu_count() - 1
    model_types = [ModelSVM] if fast else [ModelSVM]
    # Load train/test datasets, and cross_validation setting (cv)
    rcp_name_train = "RCP85"
    X_train, y_train, X_test, y_test, years_train, _, X_units, y_units, _, _, X_variable_names, y_variable_names, _, _ \
        = load_dataset_values('NPP_season_and_annual_season.csv', rcp_name_train, "RCP45")
    cv = get_cv(y_train, list(years_train), rcp_name_train)
    if fast:
        cv = cv[:1]
    # Run cross validation on the train set
    model_to_test_error = OrderedDict()
    for model_type in model_types:
        cross_validator = CrossValidator(model_type(), cv, niter, n_jobs)
        cross_validator.fit(X_train, y_train)
        error = cross_validator.error(X_test, y_test)
        print(model_type.__name__, error)

def get_cv(y_train: np.ndarray, years_train: Optional[list[int]] = None, rcp_name_train: Optional[str] = None):
    cv = []
    for validation_split, validation_size in product(validation_splits, validation_sizes):
        validation_mask = get_validation_mask(y_train, validation_size, validation_split, years_train, rcp_name_train, ['HIST', 'RCP85'])
        indices = np.arange(len(validation_mask))
        cv.append((indices, indices[validation_mask]))
    return cv

if __name__ == '__main__':
    main_cross_validation(fast=False)