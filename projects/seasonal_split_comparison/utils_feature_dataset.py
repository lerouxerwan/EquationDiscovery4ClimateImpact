from copy import deepcopy
from typing import Any, Generator

import numpy as np
from matplotlib import pyplot as plt

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.npp_season_v1 import dataset_npp_season_v1


def get_feature_datasets(original_dataset: Dataset) -> Generator[Dataset, Any, None]:
    """Generator of toy datasets, where for each dataset the target is composed/calculated from features"""
    physical_variables = set([variable_name.split('_')[0] for variable_name in original_dataset.X_variables_names])
    for physical_variable in physical_variables:
        yield get_feature_dataset(original_dataset, physical_variable)

def get_feature_dataset(original_dataset: Dataset, physical_variable: str) -> Dataset:
    """Create a toy dataset, where one physical variable, as feature in the original dataset, is set as target
    For instance, if the physical_variable = 'MLD', then the new target is mean(MLD_DJF, MLD_MAM, MLD_JJA, MLD_SON)"""
    feature_dataset = deepcopy(original_dataset)
    feature_indexes = [i for i, variable_name in enumerate(feature_dataset.X_variables_names) if variable_name.startswith(physical_variable)]
    assert len(feature_indexes) == 4, feature_indexes
    # Compute new target
    feature_dataset.y_test = np.mean(feature_dataset.X_test[:, feature_indexes], axis=1)
    feature_dataset.y_train = np.mean(feature_dataset.X_train[:, feature_indexes], axis=1)
    any_feature_index = feature_indexes[0]
    feature_dataset.y_units = [feature_dataset.X_units[any_feature_index]]
    feature_dataset.y_variable_names = [feature_dataset.X_variables_names[any_feature_index].split('_')[0]]
    label = feature_dataset.X_labels[any_feature_index]
    beginning, unit = label.split('(')
    label = beginning.split(' in ')[0] + ' (' + unit
    feature_dataset.y_labels = [label]
    # Update features (remove these indexes from several list)
    set_feature_indexes = set(feature_indexes)
    other_features_indexes = [i for i in range(len(feature_dataset.X_variables_names)) if i not in set_feature_indexes]
    feature_dataset.X_train = feature_dataset.X_train[:, other_features_indexes]
    feature_dataset.X_test = feature_dataset.X_test[:, other_features_indexes]
    feature_dataset.X_variables_names = [v for i, v in enumerate(feature_dataset.X_variables_names) if i not in set_feature_indexes]
    feature_dataset.X_units = [v for i, v in enumerate(feature_dataset.X_units) if i not in set_feature_indexes]
    feature_dataset.X_labels = [v for i, v in enumerate(feature_dataset.X_labels) if i not in set_feature_indexes]
    # Some checks and display
    feature_dataset.check()
    print(str(feature_dataset))
    return feature_dataset

if __name__ == '__main__':
    old_dataset = dataset_npp_season_v1
    for i, new_dataset in enumerate(get_feature_datasets(old_dataset)):
        new_dataset.plot_values_target(plt.gca())
        plt.show()
        if i == 1:
            break
