import sys

from projects.optimization.utils_worflow import workflow
from projects.paper.utils_paper import filename_dataset_paper


def main_workflow_slurm():
        indices = [int(s) for s in sys.argv[1:]]
        scaling_factor = [1.5, 2.][indices[0]]


        params_emulator = {
            "maxsize": 20,
            "model_selection": "custom",
            "optimizer_f_calls_limit": 10_000,
            "population_size": 31,
            "unary_operators": ["square", "sqrt"],
        }
        params_search = {
            "scaling_factor": scaling_factor,
            "n_jobs": -1,
            "n_iter": 1000,
            'param_list_to_optimize': ['populations', 'niterations', 'fraction_replaced_hof',
                                       "adaptive_parsimony_scaling", "ncycles_per_iteration",
                                       "fraction_replaced", "weight_add_node",
                                       "weight_insert_node", "weight_delete_node",
                                       "weight_do_nothing", "weight_mutate_constant",
                                       "weight_mutate_operator", "weight_swap_operands",
                                       "weight_rotate_tree", "weight_randomize",
                                       "weight_simplify", "crossover_probability",
                                       "topn", "optimizer_nrestarts",
                                       "optimizer_f_calls_limit", "optimize_probability",
                                       "optimizer_iterations", "perturbation_factor",
                                       "probability_negate_constant"]
        }
        workflow(filename_dataset_paper, params_emulator, params_search)


if __name__ == '__main__':
    main_workflow_slurm()


