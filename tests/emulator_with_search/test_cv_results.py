import numpy as np

from data.utils_dataset.npp_season_v1 import dataset_npp_season_v1
from emulator_with_search.pysr_emulator_with_search import PySREmulatorWithSearch
from emulator_with_search.utils_cv_results.utils_df_results import RMSE_VALIDATION_COLUMN_NAME, \
    PARAMS_EMULATOR_COLUMN_NAME, SELECTED_COMPLEXITY_COLUMN_NAME, get_selected_feature_indexes, \
    SELECTED_FEATURE_INDEXES_COLUMN_NAME
from tests.emulator.utils_tests_emulator import run_three_main_functions_with_one_feature


def test_df_cv_results():
    # Run validation with the hyperparameter 'populations' that can have 2 values sampled between 10 and 20
    n_iter = 2
    emulator = PySREmulatorWithSearch(niterations=1, param_grid={'populations': [10, 20]}, n_iter=n_iter)
    run_three_main_functions_with_one_feature(emulator)
    # Check number of lines in df
    df = emulator.search_experiment_.df_cv_results
    assert len(df) == n_iter
    # Check the best selected complexity
    assert df[SELECTED_COMPLEXITY_COLUMN_NAME].values[0] == 9
    # Check the selected feature indexes
    assert df[SELECTED_FEATURE_INDEXES_COLUMN_NAME].values[0] == [0]
    # Check that it is well ranked
    validation_rmse_sorted_values = df[RMSE_VALIDATION_COLUMN_NAME].values
    for rmse1, rmse2 in zip(validation_rmse_sorted_values[:-1], validation_rmse_sorted_values[1:]):
        if not np.isnan(rmse2):
            assert rmse1 <= rmse2
    # Check that the best params are as expected
    best_params = df[PARAMS_EMULATOR_COLUMN_NAME].values[0]
    assert isinstance(best_params, dict)
    assert best_params['populations'] == 19
    assert best_params['niterations'] == 1
    # Remove folders at the end of the test
    emulator.search_experiment_.remove_folder()

def test_selected_feature_indexes():
    selected_variable_names = ['x0', 'x10', 'x84']
    # One test with variable_names = None
    assert get_selected_feature_indexes(selected_variable_names) == [0, 10, 84]
    # One test with specified variable_names
    variables_names = [f'x{2 * i}' for i in range(50)]
    assert get_selected_feature_indexes(selected_variable_names, variables_names) == [0, 5, 42]

def test_custom_model_selection():
    dataset = dataset_npp_season_v1
    params = {'adaptive_parsimony_scaling': np.float64(816.4197141003457), 'alpha': 3.17, 'annealing': False, 'autodiff_backend': None, 'batch_size': 50, 'batching': False, 'binary_operators': None, 'bumper': False, 'cluster_manager': None, 'complexity_mapping': None, 'complexity_of_constants': None, 'complexity_of_operators': None, 'complexity_of_variables': None, 'constraints': None, 'crossover_probability': np.float64(0.03645040661041819), 'data_augmentation_ratio': 1, 'data_augmentation_sigma': 1.0, 'delete_tempfiles': True, 'denoise': False, 'deterministic': True, 'dimensional_constraint_penalty': 100000000, 'dimensionless_constants_only': False, 'early_stop_condition': None, 'elementwise_loss': None, 'expression_spec': None, 'extra_jax_mappings': None, 'extra_sympy_mappings': None, 'extra_torch_mappings': None, 'fast_cycle': False, 'fraction_replaced': np.float64(0.0004429864911833153), 'fraction_replaced_hof': np.float64(0.09963726885503711), 'heap_size_hint_in_bytes': None, 'hof_migration': True, 'input_stream': 'stdin', 'logger_spec': None, 'loss_function': None, 'max_evals': None, 'maxdepth': None, 'maxsize': 20, 'migration': True, 'model_selection': 'custom', 'ncycles_per_iteration': 472, 'nested_constraints': None, 'niterations': 109, 'optimize_probability': np.float64(0.07970689105408878), 'optimizer_algorithm': 'BFGS', 'optimizer_f_calls_limit': 8324, 'optimizer_iterations': 8, 'optimizer_nrestarts': 1, 'output_directory': None, 'output_jax_format': False, 'output_torch_format': False, 'parallelism': 'serial', 'parsimony': 0.0, 'perturbation_factor': np.float64(0.09045990289487015), 'population_size': 31, 'populations': 59, 'precision': 32, 'print_precision': 5, 'probability_negate_constant': np.float64(0.006406594415059639), 'procs': None, 'progress': True, 'random_state': 42, 'run_id': None, 'select_k_features': None, 'should_optimize_constants': True, 'should_simplify': True, 'skip_mutation_failures': True, 'temp_equation_file': True, 'tempdir': None, 'threshold_for_model_selection': 1.0397769088385496, 'timeout_in_seconds': None, 'topn': 20, 'tournament_selection_n': 18, 'tournament_selection_p': 0.982, 'turbo': False, 'unary_operators': ['square', 'sqrt'], 'update': False, 'update_verbosity': None, 'use_frequency': True, 'use_frequency_in_tournament': True, 'verbosity': 0, 'warm_start': False, 'warmup_maxsize_by': None, 'weight_add_node': np.float64(3.716987016612159), 'weight_delete_node': np.float64(0.8731863557229343), 'weight_do_nothing': np.float64(0.3037130122259829), 'weight_insert_node': np.float64(0.011084426436652957), 'weight_mutate_constant': np.float64(0.022677443556439466), 'weight_mutate_operator': np.float64(0.39883855941752566), 'weight_optimize': 0.0, 'weight_randomize': np.float64(0.00037043799727333546), 'weight_rotate_tree': np.float64(2.2030242310409216), 'weight_simplify': np.float64(0.0025569918001819283), 'weight_swap_operands': np.float64(0.1265512527625602), 'weighted_loss_ratio': 1.0}
    emulator = PySREmulatorWithSearch(**params, n_iter=1, scaling_factor=0, load_search_experiment=False)
    emulator.fit(dataset.X_train, dataset.y_train, validation_mask=dataset.validation_mask,
                 variable_names=dataset.X_variables_names, X_units=dataset.X_units, y_units=dataset.y_units)
    assert emulator.selected_complexity == 16

def test_custom_model_selection_version_2():
    dataset = dataset_npp_season_v1
    params =  {"maxsize": 20, "optimizer_f_calls_limit": 10000, "population_size": 31}
    emulator = PySREmulatorWithSearch(**params, n_iter=1, scaling_factor=0, load_search_experiment=False)
    emulator.fit(dataset.X_train, dataset.y_train, validation_mask=dataset.validation_mask,
                 variable_names=dataset.X_variables_names, X_units=dataset.X_units, y_units=dataset.y_units)
    assert emulator.selected_complexity == 7
