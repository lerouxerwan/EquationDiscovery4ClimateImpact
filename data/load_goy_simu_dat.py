from pathlib import Path

import pandas as pd

from utils.utils_path import DATA_PATH


def load_goy_simu_dat(nb_variables: int, simu_id: int) -> pd.DataFrame:
    """Returns a Dataframe with X rows containing nb_variables values
    where X is the number of simulation steps"""
    assert nb_variables in [10, 22]
    assert simu_id in [1, 2, 3]
    filepath_simu_dat = Path(DATA_PATH) / f'GOY/Simu_N{nb_variables}_{simu_id}/data.dat'

    data = []
    with open(filepath_simu_dat, 'r') as file:
        for line_id, line in enumerate(file):
            s = line.strip()
            row = [float(v) for v in s.split()[::2]]
            data.append(row)
            if line_id > 9:
                break
    df = pd.DataFrame(data, columns=[f'x{i}' for i in range(len(row))])
    # assert df.shape == (1_000_000, nb_variables)
    return df

if __name__ == '__main__':
    for nb_variables in [10, 22]:
        for simu_id in [1, 2, 3]:
            load_goy_simu_dat(nb_variables, simu_id)
            print(f'ok {nb_variables} {simu_id}')