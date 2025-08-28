import numpy as np
from pysr import PySRRegressor

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.npp_season_v1 import dataset_npp_season_v1
from data.utils_dataset.utils_validation import get_X_and_y
from dataset import variable_names, X_units, y_units, X, y


def main():

    params = {'model_selection': 'best', 'adaptive_parsimony_scaling': np.float64(816.4197141003457), 'alpha': 3.17,
          'annealing': False, 'autodiff_backend': None, 'batch_size': 50, 'batching': False, 'binary_operators': None,
          'bumper': False, 'cluster_manager': None, 'complexity_mapping': None, 'complexity_of_constants': None,
          'complexity_of_operators': None, 'complexity_of_variables': None, 'constraints': None,
          'crossover_probability': np.float64(0.03645040661041819), 'denoise': False, 'deterministic': True,
          'dimensional_constraint_penalty': 100000000, 'dimensionless_constants_only': False,
          'early_stop_condition': None, 'elementwise_loss': None, 'expression_spec': None, 'extra_jax_mappings': None,
          'extra_sympy_mappings': None, 'extra_torch_mappings': None, 'fast_cycle': False,
          'fraction_replaced': np.float64(0.0004429864911833153),
          'fraction_replaced_hof': np.float64(0.09963726885503711), 'heap_size_hint_in_bytes': None,
          'hof_migration': True, 'input_stream': 'stdin', 'logger_spec': None, 'loss_function': None, 'max_evals': None,
          'maxdepth': None, 'maxsize': 20, 'migration': True, 'ncycles_per_iteration': 472, 'nested_constraints': None,
          'niterations': 109, 'optimize_probability': np.float64(0.07970689105408878), 'optimizer_algorithm': 'BFGS',
          'optimizer_f_calls_limit': 8324, 'optimizer_iterations': 8, 'optimizer_nrestarts': 1,
          'output_directory': None, 'output_jax_format': False, 'output_torch_format': False, 'parallelism': 'serial',
          'parsimony': 0.0, 'perturbation_factor': np.float64(0.09045990289487015), 'population_size': 31,
          'populations': 59, 'precision': 64, 'print_precision': 5,
          'probability_negate_constant': np.float64(0.006406594415059639), 'procs': None, 'progress': True,
          'random_state': 42, 'run_id': None, 'select_k_features': None, 'should_optimize_constants': True,
          'should_simplify': True, 'skip_mutation_failures': True, 'tempdir': None, 'timeout_in_seconds': None,
          'topn': 20, 'tournament_selection_n': 18, 'tournament_selection_p': 0.982, 'turbo': False,
          'unary_operators': ['square', 'sqrt'], 'update': False, 'update_verbosity': None, 'use_frequency': True,
          'use_frequency_in_tournament': True, 'verbosity': 0, 'warm_start': False, 'warmup_maxsize_by': None,
          'weight_add_node': np.float64(3.716987016612159), 'weight_delete_node': np.float64(0.8731863557229343),
          'weight_do_nothing': np.float64(0.3037130122259829), 'weight_insert_node': np.float64(0.011084426436652957),
          'weight_mutate_constant': np.float64(0.022677443556439466),
          'weight_mutate_operator': np.float64(0.39883855941752566), 'weight_optimize': 0.0,
          'weight_randomize': np.float64(0.00037043799727333546), 'weight_rotate_tree': np.float64(2.2030242310409216),
          'weight_simplify': np.float64(0.0025569918001819283), 'weight_swap_operands': np.float64(0.1265512527625602)}
    model = PySRRegressor(**params)

    # model.fit(X, y, variable_names=variable_names, X_units=X_units, y_units=y_units)
    dataset: Dataset = dataset_npp_season_v1
    X_fit, y_fit = get_X_and_y(dataset.X_train, dataset.y_train, dataset.validation_mask, validation_set=False)

    model.fit(X_fit, y_fit, variable_names=dataset.X_variables_names, X_units=dataset.X_units, y_units=dataset.y_units)

    print(len(model.equations_))
    print(model.equations_['complexity'].to_list())
    print(model.equations_['loss'].to_list())
    print(model.equations_['sympy_format'].to_list())

if __name__ == '__main__':
    main()