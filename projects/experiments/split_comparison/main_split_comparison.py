import os
import os.path as op
from collections import OrderedDict

import numpy as np
import pandas as pd

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit, validation_split_to_validation_name
from plot.utils_plot import plot_diagnosis
from emulator.emulator_validated_with_search import EmulatorValidatedWithSearch
from projects.experiments.split_comparison.utils_feature_dataset import get_feature_datasets
from projects.utils_params import get_params_search, get_params_emulator
from utils.utils_latex import print_df_latex
from utils.utils_log import log_info


def main_split_comparison(show: bool = False, fast: bool = False):
    # Select and check validation split
    niter = 20
    validation_splits = [ValidationSplit.RCP_START, ValidationSplit.END,
                         ValidationSplit.START, ValidationSplit.SYMMETRICAL,
                         ValidationSplit.EXTREME][:1]
    # Start loop
    all_series_rmse = []
    all_series_absolute_percentage = []
    all_series_equations = []
    for validation_split in validation_splits:
        df_true, df_predict, df_infos = compute_dataframes(validation_split, niter)
        df_squared_error = (df_true - df_predict)**2
        series_rmse = df_squared_error.mean(axis=0).apply(np.sqrt)
        series_absolute_percentage = (100 * (df_predict - df_true) / df_true).apply(np.abs).mean(axis=0)
        validation_name = validation_split_to_validation_name[validation_split]
        series_rmse.name = validation_name
        all_series_rmse.append(series_rmse)
        series_equation = df_infos.iloc[0]
        series_equation.name = validation_name
        all_series_equations.append(series_equation)
        all_series_absolute_percentage.append(series_absolute_percentage)

    # Compute df_rmse
    df_rmse = pd.concat(all_series_rmse, axis=1)
    # Compute df_absolute_percentages
    df_absolute_percentages = pd.concat(all_series_absolute_percentage, axis=1)
    # Compute df_ranks
    df_ranks_rmse = df_rmse.rank(axis=1)
    df_ranks_rmse.loc['Mean rank'] = df_ranks_rmse.mean()
    # Rounds dataframes
    df_rmse = df_rmse.round(decimals=2)
    df_absolute_percentages = df_absolute_percentages.round(decimals=2)
    df_ranks_rmse = df_ranks_rmse.round(decimals=1)
    # Create a column with
    df_absolute_percentages['Mean'] = df_absolute_percentages.mean(axis=1)
    df_absolute_percentages = df_absolute_percentages.sort_values(by="Mean", axis=0)
    df_ranks_rmse = df_ranks_rmse.loc[df_absolute_percentages.index]
    # Print all dataframes
    for df in [df_ranks_rmse, df_absolute_percentages]:
        df = df.astype(str).replace(r'\.0$', '', regex=True)
        df.index.name = 'variable'
        df = df.reset_index()
        print_df_latex(df)

    # Compute df_equation
    # df_equation = pd.concat(all_series_equations, axis=1)
    # for i in [5, -6, -1]:
    # for i in [6, 7]:
    #     print(df_equation.iloc[i, 0])
    #     print(df_equation.iloc[i, 1])
    #     print('\n')


def compute_dataframes(validation_split, niter) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_name = validation_split_to_validation_name[validation_split]
    log_info(f"Compute dataframes for validation_split {validation_name}")
    folder = f'runs/{niter}_{validation_name}'
    true_csv_filename = folder + '/true.csv'
    predict_csv_filename =  folder + '/predict.csv'
    infos_csv_filename =  folder + '/infos.csv'

    if not op.exists(true_csv_filename):
        dataset = Dataset("NPP_season.csv", "RCP85", "RCP45", 0.3, validation_split)
        physical_variable_name_to_y_test_true = OrderedDict()
        physical_variable_name_to_y_test_predict = OrderedDict()
        physical_variable_name_to_infos = OrderedDict()
        for i, feature_dataset in enumerate(get_feature_datasets(dataset)):
            physical_variable_name = feature_dataset.y_variable_names[0]
            y_test_true, y_test_predict, infos = get_res(feature_dataset, get_params_emulator(), get_params_search(niter), folder)
            physical_variable_name_to_y_test_predict[physical_variable_name] = y_test_predict
            physical_variable_name_to_infos[physical_variable_name] = infos
            physical_variable_name_to_y_test_true[physical_variable_name] = y_test_true
        # Save dataframes
        df_true = pd.DataFrame.from_dict(physical_variable_name_to_y_test_true)
        df_true.to_csv(true_csv_filename)
        df_predict = pd.DataFrame.from_dict(physical_variable_name_to_y_test_predict)
        df_predict.to_csv(predict_csv_filename)
        df_infos = pd.DataFrame.from_dict(physical_variable_name_to_infos)
        df_infos.to_csv(infos_csv_filename)
    else:
        filenames = [true_csv_filename, predict_csv_filename, infos_csv_filename]
        df_true, df_predict, df_infos = [pd.read_csv(filename, index_col=0) for filename in filenames]
    return df_true, df_predict, df_infos

def get_res(dataset, params_emulator, params_search, folder:str):
    emulator = EmulatorValidatedWithSearch(**params_emulator, **params_search)
    emulator.fit(dataset.X_train, dataset.y_train, variable_names=dataset.X_variables_names, X_units=dataset.X_units,
                 y_units=dataset.y_units, validation_mask=dataset.validation_mask)
    y_test_predict = emulator.predict(dataset.X_test)
    plot_folder = op.join(folder, dataset.y_variable_names[0])
    if not op.exists(plot_folder):
        os.makedirs(plot_folder)
    plot_diagnosis(emulator, dataset, show=False, plot_folder=plot_folder)
    infos = [f'${emulator.selected_expr}$', emulator.selected_complexity, emulator.selected_variable_names]
    return dataset.y_test, y_test_predict, infos



if __name__ == '__main__':
    b = False
    main_split_comparison(b ,b)
    # main_split_comparaison(b,b)