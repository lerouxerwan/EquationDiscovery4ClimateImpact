from data.create_goy_dataset_csv import get_filepath_dataset_csv
from data.load_goy_simu_dat import simu_id_to_rcp_name
from data.utils_dataset.dataset import Dataset
import os.path as op

from data.utils_dataset.validation_split import ValidationSplit

def get_goy_dataset(nb_variables: int, with_validation: bool):
    csv_filepath = get_filepath_dataset_csv(nb_variables, with_validation)
    csv_filename = op.basename(csv_filepath)
    dataset = Dataset(csv_filename,
                      rcp_name_train=simu_id_to_rcp_name[1],
                      rcp_name_test=simu_id_to_rcp_name[3],
                      validation_split=ValidationSplit.HIST if with_validation else ValidationSplit.NONE)
    return dataset



if __name__ == '__main__':
    get_goy_dataset(nb_variables=22, with_validation=True)


