import numpy as np

from data.utils_dataset import load_dataset_ndarray
from emulator.utils_plots.plot_by_rcp.plot_climato import _plot_climato
from projects.paper.results.train.utils_ranked_features import compute_sorted_features
from projects.paper.utils_paper import filename_dataset_paper


def main_time_series_ranked_features(select_k_features, show):
    (X_train, y_train, _, _, X_units, _, years_train, _, rcp_name_train, _, variable_names, _, nb_historical_years) = load_dataset_ndarray(filename_dataset_paper)
    sorted_importance, sorted_names = compute_sorted_features(X_train, y_train, variable_names, nb_historical_years,
                                                              select_k_features)

    for i, column_name in enumerate(sorted_names, 1):
        idx = list(variable_names).index(column_name)
        values = X_train[:, idx].copy()
        unit = X_units[idx]
        if isinstance(unit, float) and np.isnan(unit):
            unit = '-'
        label = f'{column_name.replace('_', ' ')} (${unit}$)'
        _plot_climato(values, years_train=years_train, rcp_name_train=rcp_name_train,
                      nb_historical_years=nb_historical_years, target_label=label, show=show,
                      prefix=f'Feature#{i}')

if __name__ == '__main__':
    show = False
    select_k_features = 5
    main_time_series_ranked_features(select_k_features, show)
