import os.path as op
from pathlib import Path

import pandas as pd

from data.load_goy_simu_dat import load_goy_simu_dat
from utils.utils_path import DATA_PATH

VERSION_GOY_DATASET = 1

def get_filepath_dataset_csv(nb_variables: int):
    return Path(DATA_PATH) / 'dataset' / f'GOY_N{nb_variables}_v{VERSION_GOY_DATASET}.csv'


def main_create_goy_dataset(nb_variables: int):
    filepath_dataset_csv = get_filepath_dataset_csv(nb_variables)
    if not op.exists(filepath_dataset_csv):
        df_list = [load_goy_simu_dat(nb_variables, simu_id) for simu_id in [1, 2, 3][:1]]
        df = pd.concat(df_list, axis=0)
        print(df.shape)
        print(df.head())
        df.to_csv(filepath_dataset_csv)


if __name__ == '__main__':
    for nb_variables in [10, 22][:1]:
        main_create_goy_dataset(nb_variables)