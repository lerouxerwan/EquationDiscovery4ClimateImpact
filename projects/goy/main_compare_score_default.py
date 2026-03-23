import pandas as pd

from data.utils_dataset.dataset import Dataset
from emulator.emulator import Emulator
from plot.plot_diagnosis import plot_diagnosis
from projects.goy.utils_goy import get_goy_dataset
from utils.utils_log import log_info

def main_compare_score_default(nb_variables: int, with_validation: bool):
    dataset = get_goy_dataset(nb_variables, with_validation)

    d = {}
    for batch_size in [50, 100, 200]:
        values = [get_score(dataset, batch_size, niterations) for niterations in [50, 100, 200][:]]
        d[batch_size] = values
    df = pd.DataFrame.from_dict(d)
    print(df.head())

def get_score(dataset: Dataset, batch_size: int, niterations: int) -> float:
    emulator = Emulator(batching=True, niterations=niterations, batch_size=batch_size)
    emulator.fit(dataset.X_train, dataset.y_train, validation_mask=dataset.validation_mask,
                 variable_names=dataset.X_variable_names)
    return emulator.compute_selected_loss(dataset.X_test, dataset.y_test)

if __name__ == '__main__':
    main_compare_score_default(nb_variables=10, with_validation=False)