from projects.paper.utils_paper import filename_dataset_paper
from projects.utils_workflow_root import workflow_root


def main_workflow_root(fast: bool = False):
    params_emulator_root = {
        "n_iter": 1,
        "scaling_factor": 0,
        "model_selection": "custom",
    }
    if fast:
        params_emulator_root["niterations"] = 10
    # Run workflow for several number of features
    workflow_root(filename_dataset_paper, **params_emulator_root)


if __name__ == '__main__':
    fast = True
    main_workflow_root(fast)


