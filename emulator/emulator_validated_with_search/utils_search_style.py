from emulator.emulator_validated_with_search.search_cv_return_estimators import GridSearchReturnEstimators, RandomizedSearchCVReturnEstimators

search_style_to_search_cv_type = {
    'random': RandomizedSearchCVReturnEstimators,
    'grid': GridSearchReturnEstimators,
}