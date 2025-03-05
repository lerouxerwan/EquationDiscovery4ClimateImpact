from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from emulator.utils_attributes.utils_feature_selection import \
    run_feature_selection_PySR_that_returns_feature_importance
from emulator_with_search.utils_attributes.utils_validation import compute_ind_validation, get_X_and_y
from tests.emulator.utils_tests_emulator import load_climate_impact_emulator_with_search_for_test
from utils.utils_run import random_seed


def compute_sorted_features(X_train, y_train, variable_names, nb_historical_years, select_k_features):
    #  Load a fast emulator
    emulator = load_climate_impact_emulator_with_search_for_test(select_k_features=select_k_features, n_iter=1)
    ind_validation = compute_ind_validation(len(y_train), emulator.validation_size, nb_historical_years)
    X_train_train, y_train_train = get_X_and_y(X_train, y_train, ind_validation, validation_set=False)
    if isinstance(X_train_train, pd.DataFrame):
        X_train_train, y_train_train = X_train_train.values, y_train_train.values
    # Create random test similar to ClimateImpactEmulator
    random_state = check_random_state(random_seed)
    _ = random_state.randint(0, 2 ** 31 - 1)  # To have exactly the same random state as during the fit function

    selection_mask, feature_importance = run_feature_selection_PySR_that_returns_feature_importance(X_train_train,
                                                                                                    y_train_train,
                                                                                                    emulator.select_k_features,
                                                                                                    random_state)
    selected_feature_importance = [float(i) for i in feature_importance[selection_mask]]
    selected_variable_names = np.array(variable_names)[selection_mask]
    #  Sort by importance
    sorted_names, sorted_importance = zip(
        *list(sorted(zip(selected_variable_names, selected_feature_importance), key=itemgetter(1)))[::-1])
    return sorted_importance, sorted_names
