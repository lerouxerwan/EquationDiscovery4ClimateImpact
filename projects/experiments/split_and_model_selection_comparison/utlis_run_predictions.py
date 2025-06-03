import os.path as op
from collections import OrderedDict

import pandas as pd

from data.utils_dataset.dataset import Dataset
from projects.experiments.feature_dataset.utils_feature_dataset import get_all_datasets
from projects.experiments.split_and_model_selection_comparison.utils_y_test import get_y_test
from utils.utils_log import log_info
from utils.utils_path import CURRENT_PATH


def compute_dataframes_y(validation_split) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_name = str(validation_split)
    log_info(f"Compute dataframes for validation_split {validation_name}")
    folder = f'runs/{validation_name}'
    true_csv_filename = folder + '/true.csv'
    predict_custom_csv_filename =  folder + '/predict_custom.csv'
    predict_best_csv_filename =  folder + '/predict_best.csv'
    # infos_csv_filename =  folder + '/infos.csv'

    if not op.exists(true_csv_filename):
        print(CURRENT_PATH)
        raise ValueError(true_csv_filename)
        print('here')
        dataset = Dataset("NPP_season.csv", "RCP85", "RCP45", 0.3, validation_split)
        physical_variable_name_to_y_test_true = OrderedDict()
        physical_variable_name_to_y_test_predict_best = OrderedDict()
        physical_variable_name_to_y_test_predict_custom = OrderedDict()
        # physical_variable_name_to_infos = OrderedDict()
        for i, loop_dataset in enumerate(get_all_datasets(dataset)):
            physical_variable_name = loop_dataset.y_variable_names[0]
            y_test_true, y_test_predict_custom, y_test_predict_best = get_y_test(loop_dataset, folder)
            physical_variable_name_to_y_test_true[physical_variable_name] = y_test_true
            physical_variable_name_to_y_test_predict_best[physical_variable_name] = y_test_predict_best
            physical_variable_name_to_y_test_predict_custom[physical_variable_name] = y_test_predict_custom
            # if i == 1:
            #     break
            # physical_variable_name_to_infos[physical_variable_name] = infos
        # Save dataframes
        df_true = pd.DataFrame.from_dict(physical_variable_name_to_y_test_true)
        df_true.to_csv(true_csv_filename)
        df_predict_best = pd.DataFrame.from_dict(physical_variable_name_to_y_test_predict_best)
        df_predict_best.to_csv(predict_best_csv_filename)
        df_predict_custom = pd.DataFrame.from_dict(physical_variable_name_to_y_test_predict_custom)
        df_predict_custom.to_csv(predict_custom_csv_filename)
    else:
        filenames = [true_csv_filename, predict_best_csv_filename, predict_custom_csv_filename]
        df_true, df_predict_best, df_predict_custom = [pd.read_csv(filename, index_col=0) for filename in filenames]
    return df_true, df_predict_best, df_predict_custom
