from pathlib import Path

import numpy as np
import pandas as pd

from utils.utils_path import DATA_PATH

simu_id_to_index_name = {
    1: 'RCPTrainGoy',
    2: 'HIST',
    3: 'RCPTestGoy',
}

def load_goy_simu_dat(nb_variables: int, simu_id: int) -> pd.DataFrame:
    """Returns a Dataframe with X rows containing nb_variables values
    where X is the number of simulation steps"""
    assert nb_variables in [10, 22]
    assert simu_id in [1, 2, 3]
    filepath_simu_dat = Path(DATA_PATH) / f'GOY/Simu_N{nb_variables}_{simu_id}/data.dat'

    data = []
    # # Remove the first 100_000 lines, because the code does not have the time to stabilize
    # start_line_id = 100_000
    # end_line_id = None
    # with open(filepath_simu_dat, 'r') as file:
    #     for line_id, line in enumerate(file):
    #         if line_id >= start_line_id:
    #             s = line.strip()
    #             #  Keep only even columns that correspond to the real part (the odd columns are the imaginary part)
    #             row = [float(v) for v in s.split()[::2]]
    #             data.append(row)
    #             if end_line_id is not None:
    #                 assert isinstance(end_line_id, int)
    #                 if line_id > end_line_id:
    #                     break

    # Extract all data
    with open(filepath_simu_dat, 'r') as file:
        for line in file:
            s = line.strip()
            #  Keep only even columns that correspond to the real part (the odd columns are the imaginary part)
            row = [float(v) for v in s.split()[::2]]
            data.append(row)
    index_name = simu_id_to_index_name[simu_id].upper()
    index = [f'{index_name}_{i}' for i in range(len(data))]
    df = pd.DataFrame(data, index=index, columns=[f'x{i}' for i in range(len(row))])
    assert df.shape == (1_000_000, nb_variables)


    #  Compute the temporal difference di (and put it as the first column, and remove the column i)
    # D'après carlos il faut predire la difference temporelle di a partir de x0,x1,x2 ...,xi-1,xi+1,...,xn
    # et je peux prendre i=7 ou 8 quand N=22 et i=4 ou 5 quand N=10"""
    i  = 4 if nb_variables == 10 else 7
    temporal_difference_values = df.iloc[1:, i].values - df.iloc[:-1, i].values
    temporal_difference_values = np.append(temporal_difference_values, [np.nan])
    # print(len(temporal_difference_values))
    temporal_difference_i = pd.Series(index=df.index, data=temporal_difference_values)
    df.insert(0, f'd{i}', temporal_difference_i)
    assert df.shape == (1_000_000, nb_variables + 1)

    # Remove the column xi  the first 99_999 values (because the numerical simulation is not stabilized yet)
    # Remove also the last values for which the target cannot be computed.
    # In total we remove 100_000 values, and we are left with 900_000 values
    df.drop(columns=f'x{i}', inplace=True)
    df = df.iloc[99_999:-1, :]
    assert df.shape == (900_000, nb_variables)


    return df

if __name__ == '__main__':
    for nb_variables in [10, 22]:
        for simu_id in [1, 2, 3]:
            load_goy_simu_dat(nb_variables, simu_id)
            print(f'ok {nb_variables} {simu_id}')