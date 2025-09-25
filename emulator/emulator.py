import time
from datetime import timedelta
from typing import Literal, Callable, Optional, Any

import numpy as np
import pandas as pd
from pysr import PySRRegressor, AbstractExpressionSpec, AbstractLoggerSpec
from pysr.export_numpy import CallableEquation
from pysr.utils import ArrayLike
from sympy import Expr, Symbol, expand, symbols, lambdify

from data.utils_dataset.utils_validation import get_X_and_y
from data.utils_run.run import Run
from data.utils_run.utils_run import get_output_directory, get_run_id
from emulator.utils_emulator import Config
from plot.utils_metric.metric import Metric, metric_to_function
from utils.utils_log import log_info
from utils.utils_non_default_params import get_non_default_params
from utils.utils_run import random_seed




class Emulator(PySRRegressor):
    """Emulator is a variant of PySRRegressor.

    The fit method has an additional parameter: 'validation_mask' (validation_mask[i] if i in the  validation set)
    'validation_mask' makes it possible to split the fit data X and y between a train and validation set
        -on the train set, the emulator is fitted
        -on the validation set, the attribute 'index_for_validated_model_selection_' is optimized for the
        novel model_selection 'validated'. This model selection selects the equation minimizing validation error
    By default, Emulator behaves like PySR: 'validation_mask' is set to None, and 'model_selection' is set to 'best'

    -> additional attribute:
        run_: Run
            it defines a 'run_directory', depending on 'fit' inputs, where results/TensorBoard logs can be saved
        index_for_validated_model_selection_: float
            Index  to select the equation that minimizes the validation error

    -> modification of the default value for some parameters:
        -randomness is fixed for reproducibility
        -no verbose from Julia
        -hall of fame files are deleted
        -dimensional_constraint_penalty equals is set by default to 10**8 (to enforce dimension constraint)
        -logger_spec set to True (in this case, in the fit function, a more specific logger will be set)"""
    run_: Optional[Run]


    def __init__(self, model_selection: Literal["best", "accuracy", "score", "validated"] = "best", *,
                 binary_operators: list[str] | None = None, unary_operators: list[str] | None = None,
                 expression_spec: AbstractExpressionSpec | None = None, niterations: int = 100, populations: int = 31,
                 population_size: int = 27, max_evals: int | None = None, maxsize: int = 30,
                 maxdepth: int | None = None, warmup_maxsize_by: float | None = None,
                 timeout_in_seconds: float | None = None, constraints: dict[str, int | tuple[int, int]] | None = None,
                 nested_constraints: dict[str, dict[str, int]] | None = None, elementwise_loss: str | None = None,
                 loss_function: str | None = None,
                 loss_function_expression: str | None = None, loss_scale: Literal["log", "linear"] = "log",
                 complexity_of_operators: dict[str, int | float] | None = None,
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
                 warm_start: bool = False, verbosity: int = 1, update_verbosity: int | None = None,
                 print_precision: int = 5, progress: bool = True, logger_spec: AbstractLoggerSpec | None | bool = True,
                 input_stream: str = "stdin", run_id: str | None = None, output_directory: str | None = None,
                 temp_equation_file: bool = False, tempdir: str | None = None, delete_tempfiles: bool = True,
                 update: bool = False, output_jax_format: bool = False, output_torch_format: bool = False,
                 extra_sympy_mappings: dict[str, Callable] | None = None,
                 extra_torch_mappings: dict[Callable, Callable] | None = None,
                 extra_jax_mappings: dict[Callable, str] | None = None, denoise: bool = False,
                 select_k_features: int | None = None,
                 **kwargs):
        # Randomness is fixed (thus parallelism is deactivated, see PySR documentation for more details)
        if random_state is None:
            random_state = random_seed
        # Remove verbosity
        verbosity = 0
        # Ensures that deterministic is True and parallelism is "serial"
        deterministic = True
        parallelism = "serial"
        super().__init__(model_selection, binary_operators=binary_operators, unary_operators=unary_operators,
                         expression_spec=expression_spec, niterations=niterations, populations=populations,
                         population_size=population_size, max_evals=max_evals, maxsize=maxsize, maxdepth=maxdepth,
                         warmup_maxsize_by=warmup_maxsize_by, timeout_in_seconds=timeout_in_seconds,
                         constraints=constraints, nested_constraints=nested_constraints,
                         elementwise_loss=elementwise_loss, loss_function=loss_function,
                         loss_function_expression=loss_function_expression, loss_scale=loss_scale,
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
        # Change default dimensional_constraint_penalty
        if self.dimensional_constraint_penalty is None:
            self.dimensional_constraint_penalty = 10 ** 8
        # Update logger_spec if needed
        if not Config.automatic_loading_and_saving:
            self.logger_spec = None
        # Create attributes
        self.index_for_validated_model_selection_ = None
        self.run_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, validation_mask: Optional[np.ndarray[bool]] = None,
            variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
            y_units: Optional[ArrayLike[str]] = None) -> "PySRRegressor":
        """Fit the emulator for some feature X, target y, and validation_mask.
        Additional information can be specified: variable_names & units (with X_units, y_units)
        Compared to the fit method of PySR, this 'fit' method:
            -has one more argument 'validation_mask', an array of bool (None by default) defining the validation split
            -only handles np.ndarray as input for X and y
            -does not handle additional parameters of PySR (weights, Xresampled, ...)

        If validation_mask is not None, we fit the emulator on the train set (X_train_train, y_train_train)
        and compute the 'index_for_validated_model_selection' on the validation set

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
        #  Initialize self.run_, a Run object that handles all the input/output processing
        self.initialize_run(X, y, validation_mask)

        # Fit with a specific run
        return self.fit_with_run(X, y, validation_mask, variable_names, X_units, y_units, self.run_)

    def initialize_run(self, X, y, validation_mask):
        #  Some checks
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
        assert isinstance(validation_mask, np.ndarray) or validation_mask is None
        #  Run settings
        #  Set output_directory based on X,y and validation_mask.
        self.output_directory_ = self.output_directory = get_output_directory(X, y, validation_mask)
        # Set run_id based on the current parameters
        self.run_id_ = self.run_id = get_run_id(self.non_default_params)
        #  Initialize a Run object, which handles all the input/output processing
        self.run_ = Run(self.output_directory, self.run_id)

    def fit_with_run(self, X: np.ndarray, y: np.ndarray, validation_mask: Optional[np.ndarray[bool]] = None,
            variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
            y_units: Optional[ArrayLike[str]] = None, run: Run = Optional) -> "PySRRegressor":
        """Method that implement additional options compared to PySR:
            -add some potential preprocessing before the fit
            -fit with TensorBoard logging
            -update/correct small difference in the loss of the self.equations_ dataframe

        Parameters & Results
        ----------
        Same as the self.fit method"""
        # Extract X_fit and y_fit
        if validation_mask is None:
            X_fit, y_fit = X, y
        else:
            X_fit, y_fit = get_X_and_y(X, y, validation_mask, validation_set=False)

        try_loading = run.has_been_saved and Config.automatic_loading_and_saving
        if try_loading:
            try:
                emulator_from_file = self.from_file(run_directory=run.run_directory)
            except RuntimeError:
                emulator_from_file = None
        else:
            emulator_from_file = None
        # Load checkpoint if it exists, otherwise run _fit method
        if emulator_from_file is not None:
            #  Start loading from a pickle file
            log_info("Load fit from file")
            self.selection_mask_ = emulator_from_file.selection_mask_
            self.nout_ = emulator_from_file.nout_
            self.feature_names_in_ = emulator_from_file.feature_names_in_
            self.equations_ = emulator_from_file.equations_
        else:
            log_info(f'Fit with {self.non_default_params}')
            #  Fit with logging and compute its duration
            start_time = time.monotonic()
            # By default, we log with tensorboard the progress for each iteration of the run
            # See https://github.com/MilesCranmer/PySR/discussions/840 for more details on log_interval
            logging = (self.logger_spec is True)
            if logging:
                self.logger_spec = run.get_logger_spec(log_interval=1 * self.populations)
            #  Fit on the train set
            super().fit(X_fit, y_fit, variable_names=variable_names, X_units=X_units, y_units=y_units)
            if logging:
                self.logger_spec = True
            end_time = time.monotonic()
            duration = str(timedelta(seconds=end_time - start_time))
            # Save duration and tensorboard command to file
            if Config.automatic_loading_and_saving:
                log_info(f'Save fit to file')
                run.save_fit(duration, verbose=False)

        # #  Insert some columns inside equations_ with some simplified members
        # indexes_to_simplify = list(self.equations_.index.copy()[1:])
        # new_index = indexes_to_simplify[-1] + 1
        # new_complexity = 31
        # for index in indexes_to_simplify[8:11]:
        #     # For each equation we compute a simplification of it
        #     nb_terms_to_simplify = 2
        #     simplified_expr = sum(expand(self.equations_.loc[index, 'sympy_format']).args[:-nb_terms_to_simplify])
        #     f = CallableEquation(simplified_expr, symbols(' '.join(variable_names)))
        #     loss = metric_to_function[Metric.MSE](y_true=y_fit, y_pred=f(X_fit))
        #     d = {
        #         'sympy_format': simplified_expr,
        #         'lambda_format': f,
        #         'loss': loss,
        #         'score': None,
        #         'equation': str(simplified_expr),
        #         'complexity': new_complexity,
        #     }
        #     new_series = pd.DataFrame(index=[new_index], columns=self.equations_.columns,
        #                               data={k: [v] for k,v in d.items()})
        #     self.equations_ = pd.concat([self.equations_, new_series])
        #     new_index += 1
        #     new_complexity += 2
        # # Sort equations_ by complexity
        # self.equations_.sort_values(by='complexity', inplace=True)



        #  Add a 'validation_loss' column in self.equations_
        if validation_mask is not None:
            X_validation, y_validation = get_X_and_y(X, y, validation_mask, validation_set=True)
            self.equations_['validation_loss'] = self.compute_loss_list(X_validation, y_validation)
            #  Set the index for the 'validated' model selection using the validation set
            self.index_for_validated_model_selection_ = np.nanargmin(self.validation_loss_list)

        return self

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
    def selected_loss(self):
        return self.selected_row['loss']

    @property
    def selected_validation_loss(self) -> float:
        return self.selected_row['validation_loss']

    @property
    def selected_validation_rmse(self) -> float:
        return np.sqrt(self.selected_validation_loss)

    @property
    def expr_list(self) -> list[Expr]:
        """List of sympy expressions for the equations of the Pareto front"""
        return self.equations_['sympy_format'].to_list()

    @property
    def selected_expr(self) -> Expr:
        """Sympy expressions for the selected equation"""
        expr = self.selected_row['sympy_format']
        # print(expand(expr).args)
        # expr = sum(expand(expr).args[:-3])
        return expr

    @property
    def selected_complexity(self) -> int:
        """Complexity for the selected equation"""
        return self.selected_row['complexity']

    @property
    def selected_variable_names(self) -> list[str]:
        """List of variables names in the selected equation"""
        return [str(s) for s in self.selected_expr.atoms(Symbol)]

    """Model/equation selection"""

    @property
    def selected_row(self) -> pd.Series:
        """Selected row/pd.Series from the Dataframe self.equations_"""
        return self.get_best()

    def get_best(self, index: int | list[int] | None = None) -> pd.Series | list[pd.Series]:
        """Compute a Series (or list of Series) representing the selected equations (complexity, loss, ...)
         If index=None, then the equation is selected using self.model_selection"""
        if (index is None) and (self.model_selection == 'validated'):
            assert self.index_for_validated_model_selection_ is not None
            index = self.index_for_validated_model_selection_
        return super().get_best(index)

    """Other changes"""

    @property
    def non_default_params(self) -> dict[str, Any]:
        return get_non_default_params(self.get_params(), type(self))

    def __repr__(self) -> str:
        """If we do not override this method, then the __repr__ method from PySR fails
        because it does not handle model_selection='validated'"""
        pass

    def remove_folder(self):
        self.run_.remove_folder()

