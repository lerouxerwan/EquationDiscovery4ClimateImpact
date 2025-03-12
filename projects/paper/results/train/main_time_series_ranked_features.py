import numpy as np

from data.utils_dataset.utils_dataset import load_dataset
from emulator.utils_plots.plot_by_rcp.plot_climato import _plot_climato
from projects.paper.results.train.utils_ranked_features import compute_sorted_features
from projects.paper.utils_paper import filename_dataset_paper


def main_time_series_ranked_features(select_k_features, show):
    (X_train, y_train, X_test, y_test, X_units, _, years_train, years_test, rcp_name_train, rcp_name_test, variable_names, _, ind_validation) = load_dataset(filename_dataset_paper)
    sorted_importance, sorted_names = compute_sorted_features(X_train, y_train, variable_names, ind_validation,
                                                              select_k_features)

    for i, column_name in enumerate(sorted_names, 1):
        idx = list(variable_names).index(column_name)
        feature_values_train = X_train[:, idx].copy()
        feature_values_test = X_test[:, idx].copy()
        unit = X_units[idx]
        if isinstance(unit, float) and np.isnan(unit):
            unit = '-'
        label = f'{column_name.replace('_', ' ')} (${unit}$)'
        _plot_climato(feature_values_train, feature_values_test, years_train, years_test, rcp_name_train, rcp_name_test,
                      nb_historical_years=ind_validation, target_label=label, show=show,
                      prefix=f'Feature#{i}')

if __name__ == '__main__':
    show = False
    select_k_features = 10
    main_time_series_ranked_features(select_k_features, show)
