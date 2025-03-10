from projects.optimization.utils_worflow import workflow
from projects.paper.utils_paper import filename_dataset_paper


def main_workflow_child():
    search_path_to_start_from = 'best'
    params_emulator_child = {
        "n_iter": 2,
        "scaling_factor": 2,
    }
    workflow(filename_dataset_paper, search_path_to_start_from, **params_emulator_child)


if __name__ == '__main__':
    main_workflow_child()


