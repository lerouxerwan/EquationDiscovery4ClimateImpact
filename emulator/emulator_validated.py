import math
from typing import Literal, Callable, Optional

import numpy as np
from pysr import AbstractExpressionSpec, AbstractLoggerSpec, PySRRegressor
from pysr.utils import ArrayLike

from data.utils_dataset.utils_validation import get_X_and_y
from emulator.emulator import Emulator


class EmulatorValidated(Emulator):
    """EmulatorValidated is an extension of Emulator that:
        -split the training/fit data X and y between a train and validation set
        -fit the emulator on the train set
        -optimize the 'threshold_for_model_selection' parameter so that 'custom' model_selection
        always selects the equation minimizing validation error

    ->additional parameters:
        validation_size: float
            represent the proportion (between 0 and 1) of data to include in the validation split.
            Default is 0.3
    """

    def __init__(self, model_selection: Literal["best", "accuracy", "score", "custom"] = "custom", *,
                 binary_operators: list[str] | None = None, unary_operators: list[str] | None = None,
                 expression_spec: AbstractExpressionSpec | None = None, niterations: int = 100, populations: int = 31,
                 population_size: int = 27, max_evals: int | None = None, maxsize: int = 30,
                 maxdepth: int | None = None, warmup_maxsize_by: float | None = None,
                 timeout_in_seconds: float | None = None, constraints: dict[str, int | tuple[int, int]] | None = None,
                 nested_constraints: dict[str, dict[str, int]] | None = None, elementwise_loss: str | None = None,
                 loss_function: str | None = None, complexity_of_operators: dict[str, int | float] | None = None,
                 complexity_of_constants: int | float | None = None,
                 complexity_of_variables: int | float | list[int | float] | None = None,
                 complexity_mapping: str | None = None, parsimony: float = 0.0,
                 dimensional_constraint_penalty: float | None = None, dimensionless_constants_only: bool = False,
                 use_frequency: bool = True, use_frequency_in_tournament: bool = True,
                 adaptive_parsimony_scaling: float = 1040.0, alpha: float = 3.17, annealing: bool = False,
                 early_stop_condition: float | str | None = None, ncycles_per_iteration: int = 380,
                 fraction_replaced: float = 0.00036, fraction_replaced_hof: float = 0.0614,
                 weight_add_node: float = 2.47, weight_insert_node: float = 0.0112, weight_delete_node: float = 0.870,
                 weight_do_nothing: float = 0.273, weight_mutate_constant: float = 0.0346,
                 weight_mutate_operator: float = 0.293, weight_swap_operands: float = 0.198,
                 weight_rotate_tree: float = 4.26, weight_randomize: float = 0.000502, weight_simplify: float = 0.00209,
                 weight_optimize: float = 0.0, crossover_probability: float = 0.0259,
                 skip_mutation_failures: bool = True, migration: bool = True, hof_migration: bool = True,
                 topn: int = 12, should_simplify: bool = True, should_optimize_constants: bool = True,
                 optimizer_algorithm: Literal["BFGS", "NelderMead"] = "BFGS", optimizer_nrestarts: int = 2,
                 optimizer_f_calls_limit: int | None = None, optimize_probability: float = 0.14,
                 optimizer_iterations: int = 8, perturbation_factor: float = 0.129,
                 probability_negate_constant: float = 0.00743, tournament_selection_n: int = 15,
                 tournament_selection_p: float = 0.982, parallelism: (
                    Literal["serial", "multithreading", "multiprocessing"] | None
            ) = None, procs: int | None = None, cluster_manager: (
                    Literal["slurm", "pbs", "lsf", "sge", "qrsh", "scyld", "htc"] | None
            ) = None, heap_size_hint_in_bytes: int | None = None, batching: bool = False, batch_size: int = 50,
                 fast_cycle: bool = False, turbo: bool = False, bumper: bool = False,
                 precision: Literal[16, 32, 64] = 32, autodiff_backend: Literal["Zygote"] | None = None,
                 random_state: int | np.random.RandomState | None = None, deterministic: bool = True,
                 warm_start: bool = False, verbosity: int = 0, update_verbosity: int | None = None,
                 print_precision: int = 5, progress: bool = True, logger_spec: AbstractLoggerSpec | None = None,
                 input_stream: str = "stdin", run_id: str | None = None, output_directory: str | None = None,
                 temp_equation_file: bool = True, tempdir: str | None = None, delete_tempfiles: bool = True,
                 update: bool = False, output_jax_format: bool = False, output_torch_format: bool = False,
                 extra_sympy_mappings: dict[str, Callable] | None = None,
                 extra_torch_mappings: dict[Callable, Callable] | None = None,
                 extra_jax_mappings: dict[Callable, str] | None = None, denoise: bool = False,
                 select_k_features: int | None = None, threshold_for_model_selection: float = 1.5,
                 data_augmentation_ratio: int = 1,
                 data_augmentation_sigma: float = 1.0,
                 weighted_loss_ratio: float = 1.0,
                 # Additional parameters
                 validation_size: float = 0.3,
                 **kwargs):
        super().__init__(model_selection, binary_operators=binary_operators, unary_operators=unary_operators,
                         expression_spec=expression_spec, niterations=niterations, populations=populations,
                         population_size=population_size, max_evals=max_evals, maxsize=maxsize, maxdepth=maxdepth,
                         warmup_maxsize_by=warmup_maxsize_by, timeout_in_seconds=timeout_in_seconds,
                         constraints=constraints, nested_constraints=nested_constraints,
                         elementwise_loss=elementwise_loss, loss_function=loss_function,
                         complexity_of_operators=complexity_of_operators,
                         complexity_of_constants=complexity_of_constants,
                         complexity_of_variables=complexity_of_variables, complexity_mapping=complexity_mapping,
                         parsimony=parsimony, dimensional_constraint_penalty=dimensional_constraint_penalty,
                         dimensionless_constants_only=dimensionless_constants_only, use_frequency=use_frequency,
                         use_frequency_in_tournament=use_frequency_in_tournament,
                         adaptive_parsimony_scaling=adaptive_parsimony_scaling, alpha=alpha, annealing=annealing,
                         early_stop_condition=early_stop_condition, ncycles_per_iteration=ncycles_per_iteration,
                         fraction_replaced=fraction_replaced, fraction_replaced_hof=fraction_replaced_hof,
                         weight_add_node=weight_add_node, weight_insert_node=weight_insert_node,
                         weight_delete_node=weight_delete_node, weight_do_nothing=weight_do_nothing,
                         weight_mutate_constant=weight_mutate_constant, weight_mutate_operator=weight_mutate_operator,
                         weight_swap_operands=weight_swap_operands, weight_rotate_tree=weight_rotate_tree,
                         weight_randomize=weight_randomize, weight_simplify=weight_simplify,
                         weight_optimize=weight_optimize, crossover_probability=crossover_probability,
                         skip_mutation_failures=skip_mutation_failures, migration=migration,
                         hof_migration=hof_migration, topn=topn, should_simplify=should_simplify,
                         should_optimize_constants=should_optimize_constants, optimizer_algorithm=optimizer_algorithm,
                         optimizer_nrestarts=optimizer_nrestarts, optimizer_f_calls_limit=optimizer_f_calls_limit,
                         optimize_probability=optimize_probability, optimizer_iterations=optimizer_iterations,
                         perturbation_factor=perturbation_factor,
                         probability_negate_constant=probability_negate_constant,
                         tournament_selection_n=tournament_selection_n, tournament_selection_p=tournament_selection_p,
                         parallelism=parallelism, procs=procs, cluster_manager=cluster_manager,
                         heap_size_hint_in_bytes=heap_size_hint_in_bytes, batching=batching, batch_size=batch_size,
                         fast_cycle=fast_cycle, turbo=turbo, bumper=bumper, precision=precision,
                         autodiff_backend=autodiff_backend, random_state=random_state, deterministic=deterministic,
                         warm_start=warm_start, verbosity=verbosity, update_verbosity=update_verbosity,
                         print_precision=print_precision, progress=progress, logger_spec=logger_spec,
                         input_stream=input_stream, run_id=run_id, output_directory=output_directory,
                         temp_equation_file=temp_equation_file, tempdir=tempdir, delete_tempfiles=delete_tempfiles,
                         update=update, output_jax_format=output_jax_format, output_torch_format=output_torch_format,
                         extra_sympy_mappings=extra_sympy_mappings, extra_torch_mappings=extra_torch_mappings,
                         extra_jax_mappings=extra_jax_mappings, denoise=denoise, select_k_features=select_k_features,
                         threshold_for_model_selection=threshold_for_model_selection,
                         data_augmentation_ratio=data_augmentation_ratio, data_augmentation_sigma=data_augmentation_sigma,
                         weighted_loss_ratio=weighted_loss_ratio,
                         **kwargs)
        self.validation_size = validation_size
        # Some checks
        assert isinstance(self.validation_size, float) and (0 < self.validation_size < 1)


    def _fit(self, X: np.ndarray, y: np.ndarray, validation_mask: Optional[np.ndarray[bool]] = None,
            variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
            y_units: Optional[ArrayLike[str]] = None) -> "PySRRegressor":
        """Method that:
            1) fit on the train set (X_train_train, y_train_train) the emulator
            2) optimize the threshold_for_model_selection on the validation set (X_validation, y_validation)"""
        # Some check
        assert validation_mask is not None
        # Fit on the train set
        X_train, y_train = get_X_and_y(X, y, validation_mask, validation_set=False)
        super()._fit(X_train, y_train, None, variable_names, X_units, y_units)
        # Set the optimal threshold for the 'custom' model selection using the validation set
        X_validation, y_validation = get_X_and_y(X, y, validation_mask, validation_set=True)
        validation_loss_list = self.compute_loss_list(X_validation, y_validation)
        self.threshold_for_model_selection = self.compute_optimal_threshold(self.loss_list, validation_loss_list)
        return self

    @staticmethod
    def compute_optimal_threshold(train_loss_list: list[float], validation_loss_list: list[float]) -> float:
        # Compute the threshold with maximum precision
        train_loss_min = min(train_loss_list)
        index_validation_loss_min = np.nanargmin(validation_loss_list)
        train_loss_for_optimal_equation = train_loss_list[index_validation_loss_min]
        optimal_threshold = train_loss_for_optimal_equation / train_loss_min
        #  Round above (with the ceiling function) the threshold above some digits:
        # This is done to avoid issues for the custom selection
        # Otherwise due to rounding in the multiplication operation, the correct equation was sometimes not selected
        #  (because its loss value was just above min_loss_value * threshold, due to small roundings)
        nb_digits_for_upper_rounding = 10
        scaling = 10 ** nb_digits_for_upper_rounding
        optimal_threshold = float(math.ceil(optimal_threshold * scaling)) / scaling
        return optimal_threshold







