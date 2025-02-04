import os
import os.path as op

import numpy as np
import pandas as pd

from data.utils_dataset import load_dataset_dataframe
from data.utils_search import get_experiment_path, CSV_FILENAME, METRIC_COLUMN_NAME, JSON_FILENAME


def ranking(X, y, validation_size: float = 0.3, feature_selection_name: str = 'PySRDefault', select_k_features: int = 5):
    experiment_path = get_experiment_path(X, y, validation_size, feature_selection_name, select_k_features)
    param_folders = os.listdir(experiment_path)
    df_list = []
    for param_folder in param_folders:
        filepath_search_result = str(op.join(experiment_path, param_folder, CSV_FILENAME))
        df_param = pd.read_csv(filepath_search_result, index_col=0)
        df_param['param_folder'] = param_folder
        df_list.append(df_param)
    df = pd.concat(df_list, axis=0)
    df_ranked = df.sort_values(by=METRIC_COLUMN_NAME, ascending=False).drop_duplicates(subset=METRIC_COLUMN_NAME)
    # Show the top 5 equations
    nb_top_values = 5
    for i, (_, row) in list(enumerate(df_ranked.iloc[:nb_top_values].iterrows(), 1))[::-1]:
        line = f'#{i} RMSE={np.sqrt(-row[METRIC_COLUMN_NAME])} for {row["selected_expr"]}'
        print(line)
    print('\nCommand to open the JSON file that generated the best equation:')
    json_filepath = op.join(experiment_path, df.iloc[0].loc['param_folder'], JSON_FILENAME)
    print(f'cat {json_filepath}')


def main_ranking(filename):
    fast = True
    select_k_features = 3 if fast else 5
    X_train, y_train,  *_ = load_dataset_dataframe(filename)
    ranking(X_train, y_train, select_k_features=select_k_features)

if __name__ == '__main__':
    main_ranking("NPP_season.csv")