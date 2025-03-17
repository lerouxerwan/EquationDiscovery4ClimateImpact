from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from emulator_with_search.utils_search.base_search_cv_pysr import GridSearchPySR, RandomizedSearchCVPySR

search_style_to_search_cv_type = {
    'random': RandomizedSearchCVPySR,
    'grid': GridSearchPySR,
}