
import os.path as op
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from data.load_goy_simu_dat import load_goy_simu_dat
from projects.gaussian_lorenz.lorenz_additive_noise import generate_trajectories
from utils.utils_path import DATA_PATH

VERSION_LORENZ_DATASET = 1

def get_filepath_dataset_csv(state_variable_index: int, n_trajectories: int) -> str:
    filepath = Path(DATA_PATH) / 'dataset' / f'LORENZ_{state_variable_index}_{n_trajectories}_v{VERSION_LORENZ_DATASET}'
    return str(filepath) + '.csv'


def main_create_dataset_csv(state_variable_index: int = 0, n_trajectories: int = 1):
    assert n_trajectories > 0
    assert state_variable_index in {0, 1, 2}
    # Create csv dataset
    filepath_dataset_csv = get_filepath_dataset_csv(state_variable_index, n_trajectories)
    if not op.exists(filepath_dataset_csv):
        X_train_traj, X_test_traj = generate_trajectories(n_trajectories, n_trajectories)
        df_train = create_df(state_variable_index, n_trajectories, X_train_traj, True)
        df_test = create_df(state_variable_index, n_trajectories, X_test_traj, False)
        data = np.array([[f'{c} (-)'] for c in df_train.columns]).transpose
        df_label = pd.DataFrame(index=['LABEL'], columns=df_train.columns, data=data())
        df = pd.concat([df_label, df_train, df_test], axis=0)

        print(df.shape)
        print(df.head())
        df.to_csv(filepath_dataset_csv)

def create_df(state_variable_index: int, n_trajectories: int, X_traj: np.ndarray, is_train: bool):
    """Create a dataframe for several trajectories"""
    df_list = [_create_df(state_variable_index, X_traj[i, :, :]) for i in range(n_trajectories)]
    df = pd.concat(df_list, axis=0)
    bool_str = 'TRAIN' if is_train else 'TEST'
    df['index'] = [f'RCP{bool_str}LORENZ_{i}' for i in range(len(df))]
    df.set_index('index', inplace=True)
    return df


def _create_df(state_variable_index: int, X_single_traj: np.ndarray):
    """Create a dataframe for a single trajectory"""
    d = OrderedDict()
    d['y'] = X_single_traj[1:, state_variable_index]
    for i in range(3):
        d[f'x{i+1}'] = X_single_traj[:-1, i]
    df = pd.DataFrame(d)
    return df


if __name__ == '__main__':
    for state_variable_index in list(range(3))[:]:
        for n_trajectories in [4]:
            main_create_dataset_csv(state_variable_index, n_trajectories)