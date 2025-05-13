import pandas as pd

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit, validation_split_to_name
from projects.seasonal_split_comparison.utils_feature_dataset import get_feature_datasets
from projects.seasonal_split_comparison.utils_split_comparison import get_rmse_test, get_params_search, \
    get_params_emulator
from utils.utils_latex import plot_df_latex


def main_split_comparison(show: bool = False, fast: bool = False):
    # Select and check validation split
    validation_splits = [ValidationSplit.RCP_START, ValidationSplit.RANDOM]
    if fast:
        validation_splits = validation_splits[:2]
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
    df = pd.DataFrame(validation_name_to_rmse_test_list, index=physical_variable_names).transpose()
    df['Average Score'] = df.mean(axis=1)
    df = df.transpose()
    plot_df_latex(df, show)
    # Create dataframe with the rank of the rmse score



if __name__ == '__main__':
    b = True
    main_split_comparison(b ,b)