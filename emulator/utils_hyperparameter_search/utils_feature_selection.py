from enum import Enum

import numpy as np
from pysr.feature_selection import run_feature_selection
from sklearn.utils import check_random_state


class FeatureSelection(Enum):
    PySRDefault = 0
    ExpertKnowledgeMonth = 1
    ExpertKnowledgeSeason = 2

feature_selection_name_to_feature_selection = {
    'PySRDefault': FeatureSelection.PySRDefault,
    'ExpertKnowledgeMonth': FeatureSelection.ExpertKnowledgeMonth,
    'ExpertKnowledgeSeason': FeatureSelection.ExpertKnowledgeSeason,
}

expert_feature_selection_to_feature_names = {
    FeatureSelection.ExpertKnowledgeMonth: ['MLD_Apr', 'MLD_Mar', 'Shortwave_Mar', 'Shortwave_Apr', 'MLD_Feb'],
    FeatureSelection.ExpertKnowledgeSeason: ['Max_MLD_DJF' 'Max_MLD_MAM' 'Max_StratIndex_JJA',
                                                'Max_TurboclineDepth_DJF' 'Max_TurboclineDepth_MAM'],
}

def is_feature_selection(feature_selection_name: str) -> bool:
    return feature_selection_name in feature_selection_name_to_feature_selection


def get_selection_mask(X: np.ndarray, y: np.ndarray, select_k_features: int, feature_selection_name: str,
                       feature_names_in: np.ndarray[str], random_state: int | np.random.RandomState) -> np.ndarray:
    return np.array(_get_selection_mask(X, y, select_k_features, feature_selection_name, feature_names_in, random_state))

def _get_selection_mask(X: np.ndarray, y: np.ndarray, select_k_features: int, feature_selection_name: str,
                       feature_names_in: np.ndarray[str], random_state: int | np.random.RandomState) -> list[bool]:
    random_state = check_random_state(random_state)
    feature_selection = feature_selection_name_to_feature_selection[feature_selection_name]
    if feature_selection is FeatureSelection.PySRDefault:
        return list(run_feature_selection(X, y, select_k_features, random_state))
    elif feature_selection in expert_feature_selection_to_feature_names:
        assert select_k_features <= 5
        feature_names = set(expert_feature_selection_to_feature_names[feature_selection][:select_k_features])
        return [feature_name_in in feature_names for feature_name_in in feature_names_in]
    else:
        raise NotImplementedError



