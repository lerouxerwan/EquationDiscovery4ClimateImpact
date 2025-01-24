import copy
from operator import itemgetter
from typing import Literal, Callable, cast, Any

import numpy as np
import pandas as pd
from numpy import ndarray
from pysr import PySRRegressor, AbstractExpressionSpec, AbstractLoggerSpec
from pysr.denoising import multi_denoise, denoise
from pysr.utils import ArrayLike
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import _check_feature_names_in
from sympy import Expr

from emulator.utils_feature_selection.feature_selection import get_selection_mask
from emulator.utils_metric.metric import Metric, metric_to_function
from utils.utils_log import log_info
from utils.utils_run import random_seed


class ClimateImpactEmulator(PySRRegressor):
    """ClimateImpactEmulator is a variant of PySRRegressor (deterministic, no verbose, hall of fame files are deleted)
    with several additional attributes:
        threshold_for_best_model_selection : float
            Threshold to select the best equation with the 'best' model selection
            this threshold must be larger or equal to 1
            Default is 1.5 (as specified in PySR).
        feature_selection_name: str
            Name of the feature selection to use if select_k_features is not None
            Default is PySRDefault (the default feature selection used in PySR)
    and with some modification on the default value:
        dimensional_constraint_penalty equals None by default, we set it to 10**8
    """
    cache = {}

    def __init__(self, model_selection: Literal["best", "accuracy", "score"] = "best", *,
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
                 random_state: int | np.random.RandomState | None = None, deterministic: bool = False,
                 warm_start: bool = False, verbosity: int = 1, update_verbosity: int | None = None,
                 print_precision: int = 5, progress: bool = True, logger_spec: AbstractLoggerSpec | None = None,
                 input_stream: str = "stdin", run_id: str | None = None, output_directory: str | None = None,
                 temp_equation_file: bool = False, tempdir: str | None = None, delete_tempfiles: bool = True,
                 update: bool = False, output_jax_format: bool = False, output_torch_format: bool = False,
                 extra_sympy_mappings: dict[str, Callable] | None = None,
                 extra_torch_mappings: dict[Callable, Callable] | None = None,
                 extra_jax_mappings: dict[Callable, str] | None = None, denoise: bool = False,
                 select_k_features: int | None = None,
                 # Additional attributes
                 threshold_for_best_model_selection: float = 1.5,
                 feature_selection_name: str = 'PySRDefault',
                 **kwargs):
        # Some default attributes of PySRRegressor are modified
        # Verbosity is removed
        verbosity = 0
        # All files are deleted
        temp_equation_file = True
        # Randomness is fixed, see PySRRegressor documentation for more details
        deterministic = True
        random_state = random_seed
        parallelism = "serial"
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
                         **kwargs)
        self.threshold_for_best_model_selection = threshold_for_best_model_selection
        self.feature_selection_name = feature_selection_name
        assert isinstance(self.threshold_for_best_model_selection, float)
        assert self.threshold_for_best_model_selection >= 1.
        if self.dimensional_constraint_penalty is None:
            self.dimensional_constraint_penalty = 10 ** 8

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, variable_names: ArrayLike[str] | None = None,
            X_units: ArrayLike[str] | None = None, y_units: str | ArrayLike[str] | None = None,
            use_cache: bool = False) -> "PySRRegressor":
        """
        Some arguments from the fit() method of PySR, are not handled (weights, Xresampled, ...)
        We add one argument:
             use_cache: bool; whether fit results should be saved to/loaded from cache; Default is False
        """
        if use_cache:
            key = tuple(list(self.get_X_sum_and_y_sum(X, y)) + self.hash_params)
            if key in self.cache:
                log_info('Load from cache')
                (self.equations_, self.nout_, self.selection_mask_, self.julia_state_stream_,
                 self.julia_options_stream_, self.X_units_, self.y_units_, self.feature_names_in_) = self.cache[key]
            else:
                super().fit(X, y, variable_names=variable_names, X_units=X_units, y_units=y_units)
                attributes = (self.equations_.copy(), self.nout_, self.selection_mask_, self.julia_state_stream_.copy(),
                              self.julia_options_stream_.copy(), self.X_units_, self.y_units_, self.feature_names_in_.copy())
                log_info('Save to cache')
                self.cache[key] = attributes
            return self
        else:
            return super().fit(X, y, variable_names=variable_names, X_units=X_units, y_units=y_units)

    @property
    def hash_params(self) -> list[tuple[Any] | Any]:
        """All parameters except model_selection_threshold"""
        l = sorted(list(self.get_params().items()), key=itemgetter(0))
        l2 = [tuple(v) if isinstance(v, list) else v for k, v in l if k != 'threshold_for_best_model_selection']
        assert len(l2) == len(l) - 1, 'threshold_for_best_model_selection attribute must be removed from the list'
        return l2

    @property
    def complexity_list(self) -> list[int]:
        return self.equations_['complexity'].to_list()

    @property
    def score_list(self) -> list[float]:
        return self.equations_['score'].to_list()

    @property
    def expr_list(self) -> list[Expr]:
        return self.equations_['sympy_format'].to_list()

    def compute_loss(self, X, y, metric=Metric.MSE) -> list[float]:
        """Compute a loss function for every equation of the Pareto optimal set of equations"""
        loss_function = metric_to_function[metric]
        return [loss_function(y_true=y, y_pred=y_predicted) for y_predicted in self.compute_y_predicted_list(X)]

    def compute_y_predicted_list(self, X) -> list[np.ndarray]:
        """Compute predicted vector for every equation of the Pareto front"""
        return [self.predict(X, index=index) for index in range(len(self.equations_))]

    @property
    def selected_expr(self) -> Expr:
        return self.get_best()['sympy_format']

    @property
    def selected_complexity(self) -> int:
        return self.get_best()['complexity']


    def get_X_sum_and_y_sum(self, X, y) -> tuple[float, float]:
        X_sum, y_sum = (X.sum(), y.sum()) if isinstance(X, np.ndarray) else (X.values.sum(), y.values.sum())
        return float(X_sum), float(y_sum)


    def get_best(self, index: int | list[int] | None = None) -> pd.Series | list[pd.Series]:
        """
        Get best equation using `model_selection`.

        Parameters
        ----------
        index : int | list[int]
            If you wish to select a particular equation from `self.equations_`,
            give the row number here. This overrides the `model_selection`
            parameter. If there are multiple output features, then pass
            a list of indices with the order the same as the output feature.

        Returns
        -------
        best_equation : pandas.Series
            Dictionary representing the best expression found.

        Raises
        ------
        NotImplementedError
            Raised when an invalid model selection strategy is provided.
        """
        if (index is None) and (self.model_selection == "best") and (self.threshold_for_best_model_selection != 1.5):
            min_loss_train = self.equations_["loss"].min()
            max_loss_for_filter = self.threshold_for_best_model_selection * min_loss_train
            filtered_equations = self.equations_.query(f"loss <= {max_loss_for_filter}")
            index = filtered_equations["score"].idxmax()
        return super().get_best(index)

    def _pre_transform_training_data(self, X: ndarray, y: ndarray, Xresampled: ndarray | None,
                                     variable_names: ArrayLike[str],
                                     complexity_of_variables: int | float | list[int | float] | None,
                                     X_units: ArrayLike[str] | None, y_units: ArrayLike[str] | str | None,
                                     random_state: np.random.RandomState):
        # Feature selection transformation
        if self.select_k_features:
            selection_mask = get_selection_mask(
                X, y, self.select_k_features, self.feature_selection_name, self.feature_names_in_, random_state)
            X = X[:, selection_mask]

            if Xresampled is not None:
                Xresampled = Xresampled[:, selection_mask]

            # Reduce variable_names to selection
            variable_names = cast(
                ArrayLike[str],
                [
                    variable_names[i]
                    for i in range(len(variable_names))
                    if selection_mask[i]
                ],
            )

            if isinstance(complexity_of_variables, list):
                complexity_of_variables = [
                    complexity_of_variables[i]
                    for i in range(len(complexity_of_variables))
                    if selection_mask[i]
                ]
                self.complexity_of_variables_ = copy.deepcopy(complexity_of_variables)

            if X_units is not None:
                X_units = cast(
                    ArrayLike[str],
                    [X_units[i] for i in range(len(X_units)) if selection_mask[i]],
                )
                self.X_units_ = copy.deepcopy(X_units)

            # Re-perform data validation and feature name updating
            X, y = self._validate_data_X_y(X, y)
            # Update feature names with selected variable names
            self.selection_mask_ = selection_mask
            self.feature_names_in_ = _check_feature_names_in(self, variable_names)
            self.display_feature_names_in_ = self.feature_names_in_
            log_info(f"Using features {self.feature_names_in_}")

        # Denoising transformation
        if self.denoise:
            if self.nout_ > 1:
                X, y = multi_denoise(
                    X, y, Xresampled=Xresampled, random_state=random_state
                )
            else:
                X, y = denoise(X, y, Xresampled=Xresampled, random_state=random_state)

        return X, y, variable_names, complexity_of_variables, X_units, y_units






