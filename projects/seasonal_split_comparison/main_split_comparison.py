import numpy as np
import pandas as pd

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit, validation_split_to_name
from projects.seasonal_split_comparison.utils_feature_dataset import get_feature_datasets
from projects.seasonal_split_comparison.utils_split_comparison import get_rmse_test, get_params_search, \
    get_params_emulator
from utils.utils_latex import plot_df_latex, print_df_latex


def main_split_comparison(show: bool = False, fast: bool = False):
    # Select and check validation split
    validation_splits = [ValidationSplit.START, ValidationSplit.SYMMETRICAL,
                         ValidationSplit.END, ValidationSplit.RCP_START,
                         ValidationSplit.MIN, ValidationSplit.MAX,
                         ValidationSplit.EXTREME]
    # if fast:
    #     validation_splits = validation_splits[-2:]
    assert all([validation_split in validation_split_to_name for validation_split in validation_splits])
    # Start loop
    physical_variable_names = []
    validation_name_to_rmse_test_list = dict()
    for validation_split in validation_splits:
        physical_variable_names = []
        rmse_test_list = []
        dataset = Dataset("NPP_season.csv", "RCP85", "RCP45", 0.3, validation_split)
        for i, feature_dataset in enumerate(get_feature_datasets(dataset)):
            physical_variable_names.append(feature_dataset.y_variable_names[0])
            rmse_test_list.append(get_rmse_test(feature_dataset, get_params_emulator(), get_params_search(fast)))
            if fast and (i == 1):
                break
        validation_name_to_rmse_test_list[validation_split_to_name[validation_split]] = rmse_test_list
    # Create dataframe with rmse
    print('Results')
    print(physical_variable_names)
    print(validation_name_to_rmse_test_list)
    df = pd.DataFrame(validation_name_to_rmse_test_list, index=physical_variable_names).transpose()
    df['Average Score'] = df.mean(axis=1)
    df = df.transpose()
    print(df)
    print_df_latex(df)
    plot_df_latex(df, show)
    # Create dataframe with the rank of the rmse score

def main_split_comparaison(show: bool = False, fast: bool = False):
    physical_variable_names = ['BaroclinicDynamicHeight', 'ConcentrationWaterFlux']
    validation_name_to_rmse_test_list = {'start': [np.float64(0.016582332040017106), np.float64(6.51157689198415e-13)],
     'symmetrical': [np.float64(0.018516115551062116), np.float64(7.488122435181544e-11)],
     'end': [np.float64(0.01800802697743971), np.float64(4.4341390300801985e-21)],
     'start RCP': [np.float64(0.01959262735215419), np.float64(2.4710490729937018e-21)],
     'min': [np.float64(0.005476315009147966), np.float64(2.4710490729937018e-21)],
     'max': [np.float64(0.018310180321067453), np.float64(2.4710490729937018e-21)]}
    df = pd.DataFrame(validation_name_to_rmse_test_list, index=physical_variable_names).transpose()
    df['Average Score'] = df.mean(axis=1)
    df = df.transpose()
    print(df)
    df = df.reset_index()
    print_df_latex(df)
    plot_df_latex(df, show, fontsize=30)

if __name__ == '__main__':
    b = True
    # main_split_comparison(b ,b)
    main_split_comparaison(b,b)