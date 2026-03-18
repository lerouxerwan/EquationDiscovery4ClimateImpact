from data.create_goy_dataset_csv import get_filepath_dataset_csv
from data.load_goy_simu_dat import simu_id_to_index_name
from data.utils_dataset.dataset import Dataset
import os.path as op

from data.utils_dataset.validation_split import ValidationSplit

def get_goy_dataset(nb_variables: int, with_validation: bool, validation_size: float = 0.3):
    csv_filepath = get_filepath_dataset_csv(nb_variables, with_validation)
    csv_filename = op.basename(csv_filepath)
    dataset = Dataset(csv_filename,
                      rcp_name_train=simu_id_to_index_name[1].upper(),
                      rcp_name_test=simu_id_to_index_name[3].upper(),
                      validation_split=ValidationSplit.RANDOM if with_validation else ValidationSplit.NONE,
                      validation_size=validation_size)
    return dataset



if __name__ == '__main__':
    get_goy_dataset(nb_variables=22, with_validation=True)


