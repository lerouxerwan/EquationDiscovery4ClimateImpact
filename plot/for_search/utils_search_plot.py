from itertools import combinations


def get_combinations_of_search_param_names(param_grid: dict[str, list], nb_elements: int) -> list[tuple]:
    """Return combinations of nb_elements of param names in param_grid with float/int values"""
    param_names_in_param_grid = [param_name for param_name, param_value in param_grid.items()
                                 if isinstance(param_value, (int, float))]
    return list(combinations(param_names_in_param_grid, nb_elements))
