from enum import Enum
from typing import cast, Optional

import numpy as np
from numpy._typing import NDArray
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
    return np.array(_get_selection_mask_and_feature_importance(X, y, select_k_features, feature_selection_name, feature_names_in, random_state)[0])

def _get_selection_mask_and_feature_importance(X: np.ndarray, y: np.ndarray, select_k_features: int, feature_selection_name: str,
                                               feature_names_in: np.ndarray[str], random_state: int | np.random.RandomState) -> tuple[list[bool], Optional[list[float]]]:
    random_state = check_random_state(random_state)
    feature_selection = feature_selection_name_to_feature_selection[feature_selection_name]
    if feature_selection is FeatureSelection.PySRDefault:
        selection_mask, feature_importance = run_feature_selection_PySR_that_returns_feature_importance(X, y, select_k_features, random_state)
        return list(selection_mask), feature_importance
    elif feature_selection in expert_feature_selection_to_feature_names:
        assert select_k_features <= 5
        feature_names = set(expert_feature_selection_to_feature_names[feature_selection][:select_k_features])
        return [feature_name_in in feature_names for feature_name_in in feature_names_in], None
    else:
        raise NotImplementedError

def run_feature_selection_PySR_that_returns_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    select_k_features: int,
    random_state: np.random.RandomState | None = None,
) -> tuple[NDArray[np.bool_], list[float]]:
    """
    Find most important features.

    Uses a gradient boosting tree regressor as a proxy for finding
    the k most important features in X, returning indices for those
    features as output.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import SelectFromModel

    clf = RandomForestRegressor(
        n_estimators=100, max_depth=3, random_state=random_state
    )
    clf.fit(X, y)
    selector = SelectFromModel(
        clf, threshold=-np.inf, max_features=select_k_features, prefit=True
    )
    return cast(NDArray[np.bool_], selector.get_support(indices=False)), clf.feature_importances_




