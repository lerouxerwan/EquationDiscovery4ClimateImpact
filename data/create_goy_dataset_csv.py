import os.path as op
from pathlib import Path

import pandas as pd

from data.load_goy_simu_dat import load_goy_simu_dat
from utils.utils_path import DATA_PATH

VERSION_GOY_DATASET = 1

def get_filepath_dataset_csv(nb_variables: int, with_validation: bool):
    filepath = Path(DATA_PATH) / 'dataset' / f'GOY_N{nb_variables}_v{VERSION_GOY_DATASET}'
    filepath = str(filepath)
    if with_validation:
        filepath += '_withVal'
    filepath += '.csv'
    return filepath


def main_create_goy_dataset(nb_variables: int, with_validation: bool = False):
    filepath_dataset_csv = get_filepath_dataset_csv(nb_variables, with_validation)
    if not op.exists(filepath_dataset_csv):
        simu_ids = [1, 2, 3] if with_validation else [1, 3]
        df_list = [load_goy_simu_dat(nb_variables, simu_id) for simu_id in simu_ids]
        df = pd.concat(df_list, axis=0)
        print(df.shape)
        print(df.head())
        df.to_csv(filepath_dataset_csv)


if __name__ == '__main__':
    for with_validation in [False]:
            for nb_variables in [10, 22][:]:
                main_create_goy_dataset(nb_variables, with_validation)