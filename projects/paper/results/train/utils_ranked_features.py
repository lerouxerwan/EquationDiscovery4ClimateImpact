from operator import itemgetter

from data.utils_dataset import load_dataset_ndarray
from emulator.utils_hyperparameter_search.utils_feature_selection import \
    run_feature_selection_PySR_that_returns_feature_importance
from emulator.utils_hyperparameter_search.utils_validation import compute_ind_validation, get_X_and_y
from projects.paper.utils_paper import filename_dataset_paper
from tests.emulator.utils_tests_emulator import load_climate_impact_emulator_with_search_for_test
from utils.utils_run import random_seed


def compute_sorted_features(select_k_features):
    #  Load dataset
    (X_train, y_train, X_test, y_test, X_units, y_units, years_train, years_test, rcp_name_train, rcp_name_test,
     variable_names, target_label, nb_historical_years) = load_dataset_ndarray(filename_dataset_paper)
    #  Load a fast emulator
    emulator = load_climate_impact_emulator_with_search_for_test(select_k_features=select_k_features, n_iter=1)
    ind_validation = compute_ind_validation(len(y_train), emulator.validation_size, nb_historical_years)
    X_train_train, y_train_train = get_X_and_y(X_train, y_train, ind_validation, validation_set=False)
    selection_mask, feature_importance = run_feature_selection_PySR_that_returns_feature_importance(X_train_train,
                                                                                                    y_train_train,
                                                                                                    emulator.select_k_features,
                                                                                                    random_seed)
    selected_feature_importance = [float(i) for i in feature_importance[selection_mask]]
    selected_variable_names = variable_names[selection_mask]
    #  Sort by importance
    sorted_names, sorted_importance = zip(
        *list(sorted(zip(selected_variable_names, selected_feature_importance), key=itemgetter(1)))[::-1])
    return sorted_importance, sorted_names
