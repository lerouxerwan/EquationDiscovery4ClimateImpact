from projects.optimization.utils_worflow import workflow
from projects.paper.utils_paper import filename_dataset_paper


def main_workflow_root():
    params_emulator_root = {
        "n_iter": 1,
        "scaling_factor": 0,
        "model_selection": "custom",
    }
    # Run workflow for several number of features
    workflow(filename_dataset_paper, **params_emulator_root)


if __name__ == '__main__':
    main_workflow_root()


