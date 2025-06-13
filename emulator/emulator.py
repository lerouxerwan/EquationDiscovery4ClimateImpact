import math
import time
from datetime import timedelta
from typing import Literal, Callable, Optional, Any

import numpy as np
import pandas as pd
from pysr import PySRRegressor, AbstractExpressionSpec, AbstractLoggerSpec
from pysr.utils import ArrayLike
from sympy import Expr, Symbol

from data.utils_dataset.utils_validation import get_X_and_y
from data.utils_experiment.experiment import Experiment
from data.utils_experiment.utils_experiment_path import get_experiment_path
from emulator.utils_potential_contributions.utils_data_augmentation import apply_data_augmentation
from plot.utils_metric.metric import Metric, metric_to_function
from utils.utils_log import log_info
from utils.utils_non_default_params import get_non_default_params
from utils.utils_run import random_seed


class Emulator(PySRRegressor):
    """Emulator is a variant of PySRRegressor.

    The fit method has an additional parameter: 'validation_mask' (validation_mask[i] if i in the  validation set)
    'validation_mask' makes it possible to split the fit data X and y between a train and validation set
        -on the train set, the emulator is fitted
        -on the validation set, the attribute 'threshold_for_model_selection_' is optimized for the
        novel model_selection 'validated'. This model selection selects the equation minimizing validation error
    By default, Emulator behaves like PySR: 'validation_mask' is set to None, and 'model_selection' is set to 'best'

    -> additional attribute:
        experiment_: Experiment
            it defines an 'experiment path', depending on 'fit' inputs, where results/TensorBoard logs can be saved
        threshold_for_model_selection_: float
            Threshold to select an equation based on a model selection ("best", "accuracy", "score", "validated")
            this threshold must be larger or equal to 1
            Default is 1.5 (as specified in PySR).
        niterations_warmup_maxsize: int | None
            Number of iterations for warmup (slowly increase max size from a small number up to the maxsize)
            This number of iterations is when the current maxsize will reach the user-passed maxsize.
            Default is None

    -> more additional parameters for some contributions/tricks that are deactivated by default
        data_augmentation_ratio: int
            Number of times the number of datapoints augments with data augmentation
            Default is 1, i.e. no data augmentation
        data_augmentation_sigma: float
            Sigma for the noise to create new data by data augmentation
            Default is 1.

    -> modification of the default value for some parameters:
        -randomness is fixed for reproducibility
        -no verbose from Julia
        -hall of fame files are deleted
        -dimensional_constraint_penalty equals is set by default to 10**8 (to enforce dimension constraint)
        -logger_spec set to True (in this case, in the fit function, a more specific logger will be set)"""
    experiment_: Optional[Experiment]

    def __init__(self, model_selection: Literal["best", "accuracy", "score", "validated"] = "best", *,
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
                 print_precision: int = 5, progress: bool = True, logger_spec: AbstractLoggerSpec | None | bool = True,
                 input_stream: str = "stdin", run_id: str | None = None, output_directory: str | None = None,
                 temp_equation_file: bool = True, tempdir: str | None = None, delete_tempfiles: bool = True,
                 update: bool = False, output_jax_format: bool = False, output_torch_format: bool = False,
                 extra_sympy_mappings: dict[str, Callable] | None = None,
                 extra_torch_mappings: dict[Callable, Callable] | None = None,
                 extra_jax_mappings: dict[Callable, str] | None = None, denoise: bool = False,
                 select_k_features: int | None = None,
                 # Additional parameters
                 niterations_warmup_maxsize: int | None = None,
                 # Additional parameters for potential contributions (which are deactivated by default)
                 data_augmentation_ratio: int = 1,
                 data_augmentation_sigma: float = 1.0,
                 **kwargs):
        # Randomness is fixed (thus parallelism is deactivated, see PySR documentation for more details)
        if random_state is None:
            random_state = random_seed
        # Ensures that deterministic is True and parallelism is "serial"
        deterministic = True
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
        self.niterations_warmup_maxsize = niterations_warmup_maxsize
        self.data_augmentation_ratio = data_augmentation_ratio
        self.data_augmentation_sigma = data_augmentation_sigma
        # Some checks
        assert self.niterations_warmup_maxsize is None or isinstance(self.niterations_warmup_maxsize, int)
        assert isinstance(self.data_augmentation_ratio, int)
        assert isinstance(self.data_augmentation_sigma, float)
        # Avoid some cases where the Julia code of PySR crashes
        assert self.population_size > 0
        assert self.tournament_selection_n > 0
        # Change default dimensional_constraint_penalty
        if self.dimensional_constraint_penalty is None:
            self.dimensional_constraint_penalty = 10 ** 8
        # Set params
        self.set_warmup_maxsize_by()
        # Create attributes
        self.threshold_for_model_selection_ = None
        self.experiment_ = None

    def set_params(self, **params):
        super().set_params(**params)
        self.set_warmup_maxsize_by()
        return self

    def set_warmup_maxsize_by(self):
        if self.niterations_warmup_maxsize is not None:
            assert 0 <= self.niterations_warmup_maxsize <= self.niterations
            self.warmup_maxsize_by = self.niterations_warmup_maxsize / self.niterations

    def fit(self, X: np.ndarray, y: np.ndarray, validation_mask: Optional[np.ndarray[bool]] = None,
            variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
            y_units: Optional[ArrayLike[str]] = None, **kwargs) -> "PySRRegressor":
        """Fit the emulator for some feature X, target y, and validation_mask.
        Additional information can be specified: variable_names & units (with X_units, y_units)
        Compared to the fit method of PySR, this 'fit' method:
            -has one more argument 'validation_mask', an array of bool (None by default) defining the validation split
            -only handles np.ndarray as input for X and y
            -does not handle additional parameters of PySR (weights, Xresampled, ...)

        If validation_mask is not None, we fit the emulator on the train set (X_train_train, y_train_train)
        and optimize the 'threshold_for_model_selection_' on the validation set

        Parameters
        ----------
        X : ndarray, Training data of shape (n_samples, n_features).
        y : ndarray, Target values of shape (n_samples,) or (n_samples, n_targets).
        validation_mask: Optional[ndarray], validation_mask[i] indicates if the index 'i' is in the validation set
        variable_names : list[str], a list of names for the variables, rather than "x0", "x1", etc.
        X_units : list[str], a list of units for each variable in `X`.
        y_units : str | list[str], similar to `X_units`, but as a unit for the target variable, `y`.

        Returns
        -------
        self : object
            Fitted estimator"""
        log_info('Start fit emulator')
        # Some checks
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
        assert isinstance(validation_mask, np.ndarray) or validation_mask is None
        # Initialize self.experiment_ which defines where results/TensorBoard logs can be saved
        self.experiment_ = self.get_experiment(X, y, validation_mask)
        # Run self._fit method, which can be overridden in child classes, and compute its duration
        start_time = time.monotonic()
        self._fit(X, y, validation_mask, variable_names, X_units, y_units, **kwargs)
        end_time = time.monotonic()
        duration = str(timedelta(seconds=end_time - start_time))
        # Save duration and tensorboard command to file
        self.experiment_.print_and_save_fit_information(duration)
        return self

    def get_experiment(self, X: np.ndarray, y: np.ndarray, validation_mask: Optional[np.ndarray[bool]] = None) -> Experiment:
        return Experiment(get_experiment_path(X, y, validation_mask, self.non_default_params), self.model_selection)

    def set_model_selection(self, model_selection: str) -> None:
        if self.model_selection == model_selection:
            # Do nothing
            pass
        else:
            self.model_selection = model_selection
            # Reload experiment with the updated model_selection
            self.experiment_ = Experiment(self.experiment_.experiment_path, model_selection)


    def _fit(self, X: np.ndarray, y: np.ndarray, validation_mask: Optional[np.ndarray[bool]] = None,
            variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
            y_units: Optional[ArrayLike[str]] = None) -> "PySRRegressor":
        """Method that implement additional options compared to PySR:
            -add some potential preprocessing before the fit
            -fit with TensorBoard logging
            -update/correct small difference in the loss of the self.equations_ dataframe

        Parameters & Results
        ----------
        Same as the self.fit method"""
        # Potential preprocessing (data augmentation, weights computing) before the fit that are deactivate by default
        # Apply data augmentation
        if self.data_augmentation_ratio > 1:
            X, y = apply_data_augmentation(X, y, self.data_augmentation_ratio, self.data_augmentation_sigma)
        # Fit with logging
        # By default, we log with tensorboard the progress for each iteration of the experiment
        # See https://github.com/MilesCranmer/PySR/discussions/840 for more details on log_interval"""
        logging = self.logger_spec is True
        if logging:
            self.logger_spec = self.experiment_.get_logger_spec(log_interval=1 * self.populations)
        # Fit on the train set
        if validation_mask is None:
            X_fit, y_fit = X, y
        else:
            X_fit, y_fit = get_X_and_y(X, y, validation_mask, validation_set=False)
        super().fit(X_fit, y_fit, variable_names=variable_names, X_units=X_units, y_units=y_units)

        if logging:
            self.logger_spec = True
        # Update the 'loss' column in the self.equations_ dataframe
        # because it is sometimes not consistent with the predict method)
        # See https://github.com/MilesCranmer/PySR/discussions/943 for more details on this issue
        loss_list = self.compute_loss_list(X_fit, y_fit)
        self.equations_['loss'] = loss_list
        pareto_indexes = [True] + [loss_list[i] < min(loss_list[:i]) for i in range(1, len(loss_list))]
        self.equations_ = self.equations_.loc[pd.Series(pareto_indexes, index=self.equations_.index)]
        self.equations_ = self.equations_.reset_index(drop=True)
        #  Add also a 'validation_loss' column in self.equations_
        if validation_mask is not None:
            X_validation, y_validation = get_X_and_y(X, y, validation_mask, validation_set=True)
            self.equations_['validation_loss'] = self.compute_loss_list(X_validation, y_validation)
            #  Set the optimal threshold for the 'validated' model selection using the validation set
            self.threshold_for_model_selection_ = self.compute_threshold(self.loss_list, self.validation_loss_list)
        return self

    @staticmethod
    def compute_threshold(train_loss_list: list[float], validation_loss_list: list[float]) -> float:
        #  Compute the threshold with maximum precision
        train_loss_min = min(train_loss_list)
        index_validation_loss_min = np.nanargmin(validation_loss_list)
        train_loss_for_optimal_equation = train_loss_list[index_validation_loss_min]
        optimal_threshold = train_loss_for_optimal_equation / train_loss_min
        #  Round above (with the ceiling function) the threshold above some digits:
        # This is done to avoid issues for the model selection "validated"
        # Otherwise due to rounding in the multiplication operation, the correct equation was sometimes not selected
        #  (because its loss value was just above min_loss_value * threshold, due to small roundings)
        nb_digits_for_upper_rounding = 10
        scaling = 10 ** nb_digits_for_upper_rounding
        optimal_threshold = float(math.ceil(optimal_threshold * scaling)) / scaling
        return optimal_threshold

    def compute_loss_for_set(self, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool],
                             validation_set: bool, metric: Metric) -> float:
        return self.compute_loss(*get_X_and_y(X, y, validation_mask, validation_set), metric=metric)

    """Method to compute loss"""

    def compute_loss_list(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Compute a list of loss: one loss for every equation of the Pareto optimal set of equations"""
        return [self._compute_loss(y, y_predicted, self.metric) for y_predicted in self.compute_y_predicted_list(X)]

    @property
    def metric(self) -> Metric:
        if self.loss_function is None:
            return Metric.MSE
        else:
            raise NotImplementedError('this loss function does not have a corresponding metric')

    def compute_loss_list_other_metric(self, X: np.ndarray, y: np.ndarray, metric: Metric) -> list[float]:
        """Compute a list of loss: one loss for every equation of the Pareto optimal set of equations"""
        return [self._compute_loss(y, y_predicted, metric) for y_predicted in self.compute_y_predicted_list(X)]

    @staticmethod
    def _compute_loss(y_true: np.ndarray, y_predicted: np.ndarray, metric: Metric) -> float:
        """Compute loss for a given metric, if the computation raises a ValueError we return np.nan as result"""
        loss_function = metric_to_function[metric]
        try:
            return loss_function(y_true=y_true, y_pred=y_predicted)
        except ValueError:
            return  np.nan

    def compute_loss(self, X: np.ndarray, y: np.ndarray, metric: Metric) -> float:
        return self._compute_loss(y, self.predict(X), metric)

    def compute_y_predicted_list(self, X: np.ndarray) -> list[np.ndarray]:
        """Compute predicted vector for every equation of the Pareto front"""
        return [self.predict(X, index=index) for index in range(len(self.equations_))]

    """Properties"""

    @property
    def complexity_list(self) -> list[int]:
        """List of complexity for the equations of the Pareto front (thus in increasing order)"""
        return self.equations_['complexity'].to_list()

    @property
    def loss_list(self) -> list[float]:
        """List of Train loss (Mean squared error) for the equations of the Pareto front"""
        return self.equations_['loss'].to_list()

    @property
    def validation_loss_list(self) -> list[float]:
        """List of Validation loss (Mean squared error) for the equations of the Pareto front"""
        return self.equations_['validation_loss'].to_list()

    @property
    def score_list(self) -> list[float]:
        """List of score (some heuristic defined in PySR) for the equations of the Pareto front"""
        return self.equations_['score'].to_list()

    @property
    def selected_loss(self):
        return self.selected_row['loss']

    @property
    def expr_list(self) -> list[Expr]:
        """List of sympy expressions for the equations of the Pareto front"""
        return self.equations_['sympy_format'].to_list()

    @property
    def selected_expr(self) -> Expr:
        """Sympy expressions for the selected equation"""
        return self.selected_row['sympy_format']

    @property
    def selected_complexity(self) -> int:
        """Complexity for the selected equation"""
        return self.selected_row['complexity']

    @property
    def selected_variable_names(self) -> list[str]:
        """List of variables names in the selected equation"""
        return [str(s) for s in self.selected_expr.atoms(Symbol)]

    """Model/equation selection"""

    def get_best_pysr(self) -> pd.Series:
        """Compute a Series representing the selected equation (complexity, loss, ...) with the PySR heuristic"""
        model_selection = self.model_selection[:]
        self.set_model_selection('best')
        result = super().get_best()
        self.set_model_selection(model_selection)
        return result

    @property
    def selected_row(self) -> pd.Series:
        """Selected row/pd.Series from the Dataframe self.equations_"""
        return self.get_best()

    def get_best(self, index: int | list[int] | None = None) -> pd.Series | list[pd.Series]:
        """Compute a Series (or list of Series) representing the selected equations (complexity, loss, ...)
         If index=None, then the equation is selected using self.model_selection"""
        if (index is None) and (self.model_selection == 'validated'):
            # Select the index of self.equations_ with the maximum train loss (for 'validated')
            index = self.filtered_equations['loss'].idxmax()
        return super().get_best(index)

    @property
    def filtered_equations(self) -> pd.DataFrame:
        """Extract a Dataframe, containing only a subset of rows from self.equations,
        such that selected rows are equations such that loss < min_loss * self.threshold_for_model_selection_"""
        min_loss_train = self.equations_["loss"].min()
        max_loss_for_filter = self.threshold_for_model_selection_ * min_loss_train
        filtered_equations = self.equations_.query(f"loss <= {max_loss_for_filter}")
        return filtered_equations

    """Other changes"""

    @property
    def non_default_params(self) -> dict[str, Any]:
        return get_non_default_params(self)

    def __repr__(self) -> str:
        """If we do not override this method, then the __repr__ method from PySR fails
        because it does not handle model_selection='validated'"""
        pass


