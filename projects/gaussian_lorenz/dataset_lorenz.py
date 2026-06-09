import os.path as op
from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from projects.gaussian_lorenz.create_csv_dataset import get_filepath_dataset_csv


def get_dataset_lorenz(state_variable_index: int, n_trajectories: int):
    csv_filepath = get_filepath_dataset_csv(state_variable_index, n_trajectories)
    csv_filename = op.basename(csv_filepath)
    dataset = Dataset(csv_filename,
                      rcp_name_train="RCPTRAINLORENZ",
                      rcp_name_test="RCPTESTLORENZ",
                      validation_split=ValidationSplit.NONE)
    return dataset


if __name__ == '__main__':
    for state_variable_index in list(range(3))[:1]:
        dataset = get_dataset_lorenz(state_variable_index, 4)
