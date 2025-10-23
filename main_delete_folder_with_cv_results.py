import os
import os.path as op
from shutil import rmtree

from utils.utils_path import RUN_PATH


def main_delete_folder_with_cv_results():
    path_with_cv_results = []
    for folder in os.listdir(RUN_PATH):
        for subfolder in os.listdir(op.join(RUN_PATH, folder)):
            path = op.join(RUN_PATH, folder, subfolder)
            if 'cv_results.csv' in os.listdir(path):
                path_with_cv_results.append(path)
    print(len(path_with_cv_results))
    print(path_with_cv_results)
    for path in path_with_cv_results:
        rmtree(path)

if __name__ == '__main__':
    main_delete_folder_with_cv_results()