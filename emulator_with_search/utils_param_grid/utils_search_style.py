from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

search_style_to_search_cv_type = {
    'random': RandomizedSearchCV,
    'grid': GridSearchCV,
}