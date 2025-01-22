from enum import Enum

import numpy as np
import pandas as pd
from pysr.feature_selection import run_feature_selection
from sklearn.utils import check_random_state

from emulator.utils_feature_selection.utils_expert_knowledge import run_feature_selection_expert_knowledge


class FeatureSelection(Enum):
    PySRDefault = 0
    ExpertKnowledge = 1

feature_selection_name_to_feature_selection = {
    'PySRDefault': FeatureSelection.PySRDefault,
    'ExpertKnowledge': FeatureSelection.ExpertKnowledge,
}

def is_feature_selection(feature_selection_name: str) -> bool:
    return feature_selection_name in feature_selection_name_to_feature_selection


feature_selection_to_function_get_selection_mask = {
    FeatureSelection.PySRDefault: run_feature_selection,
    FeatureSelection.ExpertKnowledge: run_feature_selection_expert_knowledge,
}


def get_selection_mask(X: np.ndarray, y: np.ndarray, select_k_features: int, feature_selection_name: str,
                       feature_names_in: np.ndarray[str], random_state: int | np.random.RandomState) -> np.ndarray:
    return np.array(_get_selection_mask(X, y, select_k_features, feature_selection_name, feature_names_in, random_state))

def _get_selection_mask(X: np.ndarray, y: np.ndarray, select_k_features: int, feature_selection_name: str,
                       feature_names_in: np.ndarray[str], random_state: int | np.random.RandomState) -> list[bool]:
    random_state = check_random_state(random_state)
    feature_selection = feature_selection_name_to_feature_selection[feature_selection_name]
    if feature_selection is FeatureSelection.PySRDefault:
        return list(run_feature_selection(X, y, select_k_features, random_state))
    elif feature_selection is FeatureSelection.ExpertKnowledge:
        assert select_k_features == 4
        feature_names_to_select = {'Shortwave_Mar', 'Shortwave_Apr', 'MLD_Mar', 'MLD_Apr'}
        return [feature_name_in in feature_names_to_select for feature_name_in in feature_names_in]
    else:
        raise NotImplementedError



