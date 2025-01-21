import os.path as op

from sklearn.model_selection import RandomizedSearchCV

from utils.utils_path import SEARCH_CSV_PATH

NDIGITS = 1

def get_filepath_search(X_sum: float, y_sum: float, validation_size: float, search_cv_type: type, n_iter: int,
                        param_grid: dict, **params_fit) -> str:
    # Create filename search
    filename_search = f'{round(X_sum, NDIGITS)}_{round(y_sum, NDIGITS)}_{validation_size}_{search_cv_type.__name__}'
    if search_cv_type is RandomizedSearchCV:
        filename_search += f'_{n_iter}'
    if 'X_units' in params_fit:
        suffix = 'out' if params_fit['X_units'] is None else ''
        filename_search += f'_with{suffix}Units'
    filename_search += '_' + param_grid_signature(param_grid)
    # Return filepath search
    return op.join(SEARCH_CSV_PATH, filename_search + '.csv')

def param_grid_signature(param_grid: dict) -> str:
    efficient_param_grid = {}
    for name, values in param_grid.items():
        if len(values) > 1:
            efficient_param_grid[name[:3]] = f'{round(min(values), NDIGITS)}_{round(max(values), NDIGITS)}'
    names_sorted = [name for name in sorted(list(efficient_param_grid.keys()))]
    param_grid_signature = '_'.join([name + efficient_param_grid[name] for name in names_sorted])
    return param_grid_signature



if __name__ == '__main__':
    print(str(RandomizedSearchCV.__name__))