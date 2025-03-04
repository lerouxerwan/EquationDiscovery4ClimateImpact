import sys

from projects.paper.utils_paper import filename_dataset_paper
from projects.utils_workflow_root import workflow_root


def main_workflow_slurm():
        indices = [int(s) for s in sys.argv[1:]]
        select_k_features = indices[0]

        params_emulator = {
            "select_k_features": select_k_features,
            "n_iter": 50,
            "population_size": 31,
            'niterations': 10,
            "unary_operators": ["square", "sqrt"],
            "binary_operators": ["+", "*", "/", "-"],
            # Hyperparameter to optimize around (/2, x2) their default or specified value
            "scaling_factor": 2,
            "param_list_to_optimize_around_default": [
                "niterations",
                "adaptive_parsimony_scaling",
                "fraction_replaced_hof",
                "populations",
                "population_size"
            ]
        }
        # Run workflow for several number of features
        workflow_root(filename_dataset_paper, **params_emulator)


if __name__ == '__main__':
    main_workflow_slurm()


