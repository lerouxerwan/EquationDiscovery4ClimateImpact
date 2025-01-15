import os.path as op

from sklearn.model_selection import RandomizedSearchCV

from utils.utils_path import SEARCH_CSV_PATH


def get_filepath_search(X, y, validation_size, search_cv_type: type, n_iter, param_grid) -> str:
    # Create filename search
    filename_search = f'{X.sum()}_{y.sum()}_{validation_size}_{search_cv_type.__name__}'
    if search_cv_type is RandomizedSearchCV:
        filename_search += f'_{n_iter}'
    filename_search += '_' + param_grid_signature(param_grid)
    # Return filepath search
    return op.join(SEARCH_CSV_PATH, filename_search + '.csv')

def param_grid_signature(param_grid: dict) -> str:
    efficient_param_grid = {}
    for name, values in param_grid.items():
        if len(values) > 1:
            efficient_param_grid[name[:3]] = f'{min(values)}{max(values)}'
    names_sorted = [name for name in sorted(list(efficient_param_grid.keys()))]
    param_grid_signature = '_'.join([name + efficient_param_grid[name] for name in names_sorted])
    return param_grid_signature



if __name__ == '__main__':
    print(str(RandomizedSearchCV.__name__))