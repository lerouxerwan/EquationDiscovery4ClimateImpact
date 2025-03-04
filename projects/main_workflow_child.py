from projects.paper.utils_paper import filename_dataset_paper
from projects.utils_workflow_child import workflow_child
from projects.utils_workflow_root import workflow_root


def main_workflow(fast: bool = False):
    if fast:
        parents_search_dir = (r'/home/e23lerou/Documents/EquationDiscovery4ClimateImpact/data/search/2505811.3_3493.2_0.3'
                              r'/PySRDefault_None/RandomizedSearchCV_1_nit1_nitons10_ada1040.0_1040.0_fra0.1_0.1_nit10_10_sca0')
        params_emulator = {
            "n_iter": 5,
            "scaling_factor": 2,
            "param_list_to_optimize_around_default": ["niterations"]
        }
    else:
        params_emulator = {
        }
    # Run workflow for several number of features
    workflow_child(filename_dataset_paper, parents_search_dir, **params_emulator)


if __name__ == '__main__':
    fast = True
    main_workflow(fast)


