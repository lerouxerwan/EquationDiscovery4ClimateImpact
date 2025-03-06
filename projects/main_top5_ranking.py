import os
import os.path as op

import numpy as np
import pandas as pd

from emulator_with_search.search_folder.utils_search_folder import CSV_FILENAME, METRIC_COLUMN_NAME, JSON_FILENAME, \
    get_dataset_dir
from utils.utils_dataset import load_dataset_dataframe
from utils.utils_json_loader import JsonLoader


def ranking(X, y, validation_size: float = 0.3):
    dataset_search_path = get_dataset_dir(X, y, validation_size)
    filepath_search_results = []
    for experiment_folder in os.listdir(dataset_search_path):
        for param_folder in os.listdir(op.join(dataset_search_path, experiment_folder)):
            filepath_search_result = op.join(dataset_search_path, experiment_folder, param_folder, CSV_FILENAME)
            filepath_search_results.append(filepath_search_result)
    _ranking(filepath_search_results)



def _ranking(filepath_search_results):
    df_list = []
    for filepath_search_result in filepath_search_results:
        print(filepath_search_result)
        df_param = pd.read_csv(filepath_search_result, index_col=0)
        df_param['filepath_search_result'] = filepath_search_result
        print(float(np.sqrt(-df_param.iloc[0][METRIC_COLUMN_NAME])))
        df_list.append(df_param)
    df = pd.concat(df_list, axis=0)
    df = df.sort_values(by=METRIC_COLUMN_NAME, ascending=False).drop_duplicates(subset=METRIC_COLUMN_NAME)
    #  Show the top equations
    nb_top_values = 5
    print(f'Top {nb_top_values} Equations:\n')
    for i, (_, row) in list(enumerate(df.iloc[:nb_top_values].iterrows(), 1))[::-1]:
        rmse = float(np.sqrt(-row[METRIC_COLUMN_NAME]))
        selected_expr = row["selected_expr"]
        param_name = op.basename(op.dirname(op.dirname(row['filepath_search_result'])))
        params = JsonLoader.load(row["params"])
        _ = params.pop('threshold_for_model_selection')
        line = f'#{i} RMSE={round(rmse, 3)} for {param_name} {params} with {selected_expr}'
        print(line)
    print('\nCommand to open the JSON file that generated the best equation:')
    json_filepath = df.iloc[0].loc['filepath_search_result'].replace(CSV_FILENAME, JSON_FILENAME)
    print(f'cat {json_filepath}')


def main_ranking(filename):
    X_train, y_train,  *_ = load_dataset_dataframe(filename)
    ranking(X_train, y_train)
    # fast = False
    # select_k_features = 3 if fast else 5
    # ranking_local(X_train, y_train, select_k_features=select_k_features)

if __name__ == '__main__':
    main_ranking("NPP_season.csv")