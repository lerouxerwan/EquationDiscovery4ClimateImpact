import time
import warnings
from datetime import timedelta
from itertools import chain
from typing import Literal, Callable, Optional, Any

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas.errors import EmptyDataError
from pysr import PySRRegressor, AbstractExpressionSpec, AbstractLoggerSpec, TemplateExpressionSpec
from pysr.utils import ArrayLike
from scipy.stats import norm
from sympy import Expr, Symbol, sympify, symbols

from data.utils_dataset.utils_validation import get_X_and_y
from data.utils_run.run import Run
from data.utils_run.utils_run import get_output_directory, get_run_id, get_non_default_params
from emulator.is_interpretable import is_interpretable
from emulator.utils_gaussian_fit import get_X_for_gaussian_fit, get_lambda_function_list, get_loss_str_gaussian_fit, \
    compute_loss_gaussian_fit, UncertaintyInterval
from emulator.utils_variable_names import get_variable_names, get_variable_signed_names
from plot.by_split.utils_equation_str import replace_julia_square_by_python_power, \
    postprocessing_for_equation, get_equation_from_expr
from plot.utils_metric.metric import Metric, compute_loss
from utils.utils_log import log_info
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
                 gaussian_fit: bool = False,
                 X_variable_names_for_gaussian_fit: Optional[list[str]] = None,
                 y_variable_name_for_gaussian_fit: Optional[str] = None,
                 interpretable_mode: bool = False,
                 **kwargs):
        # Randomness is fixed (thus parallelism is deactivated, see PySR documentation for more details)
        if random_state is None:
            random_state = random_seed
        # Remove verbosity
        verbosity = 0
        # Ensures that deterministic is True and parallelism is "serial"
        deterministic = True
        parallelism = "serial"
        # Specify loss and template expression for Gaussian fit
        # we follow the trick illustrated in https://github.com/MilesCranmer/PySR/discussions/1002
        if gaussian_fit:
            assert X_variable_names_for_gaussian_fit is not None
            assert isinstance(X_variable_names_for_gaussian_fit, list)
            assert isinstance(y_variable_name_for_gaussian_fit, str)
            X_variable_names_as_string = ', '.join(X_variable_names_for_gaussian_fit)
            if elementwise_loss is None:
                elementwise_loss = "my_custom_loss(predicted, target) = predicted"
            expressions = ["mu", "log_sigma"]
            if expression_spec is None:
                expression_spec = TemplateExpressionSpec(
                    expressions=expressions,
                    variable_names=X_variable_names_for_gaussian_fit + [y_variable_name_for_gaussian_fit],
                    combine=f"""
                        mu_value = mu({X_variable_names_as_string})
                        sigma_value = exp(log_sigma({X_variable_names_as_string}))
    
                        {get_loss_str_gaussian_fit(y_variable_name_for_gaussian_fit)}
                    """)
            else:
                if isinstance(expression_spec, TemplateExpressionSpec):
                    assert expression_spec.expressions == expressions

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
        # Add parameter
        self.interpretable_mode = interpretable_mode
        self.gaussian_fit = gaussian_fit
        self.X_variable_names_for_gaussian_fit = X_variable_names_for_gaussian_fit
        self.y_variable_name_for_gaussian_fit = y_variable_name_for_gaussian_fit
        # Change default dimensional_constraint_penalty
        if self.dimensional_constraint_penalty is None:
            self.dimensional_constraint_penalty = 10 ** 8
        # Create attributes
        self.index_for_validated_model_selection_ = None
        self.run_ = None

    def fit(self, X: ndarray, y: ndarray, validation_mask: Optional[ndarray] = None,
            variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
            y_units: Optional[ArrayLike[str]] = None) -> "PySRRegressor":
        """Fit the emulator for some feature X, target y, and validation_mask.
        Additional information can be specified: variable_names & units (with X_units, y_units)
        Compared to the fit method of PySR, this 'fit' method:
            -has one more argument 'validation_mask', an array of bool (None by default) defining the validation split
            -only handles ndarray as input for X and y
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
        #  Some checks
        self.some_checks(X, y, validation_mask)

        #  Set output_directory based on X,y and validation_mask.
        self.output_directory_ = self.output_directory = get_output_directory(X, y, validation_mask)

        # Set run_id based on the current parameters
        self.run_id = self.run_id_ = self.get_run_id(self.params)

        #  Initialize a Run object, which handles all the input/output processing
        self.run_ = Run(self.output_directory, self.run_id)

        # Fit with a specific run
        return self.fit_with_run(X, y, validation_mask, variable_names, X_units, y_units, self.run_)

    def some_checks(self, X: ndarray, y: ndarray, validation_mask: Optional[ndarray] = None) -> None:
        # Some checks on X, y and validation_mask
        assert isinstance(X, ndarray) and isinstance(y, ndarray)
        assert isinstance(validation_mask, ndarray) or validation_mask is None
        if validation_mask is not None:
            assert all([isinstance(value, np.bool) for value in validation_mask])
        # Avoid some Julia crashes
        if self.population_size <= self.tournament_selection_n:
            self.tournament_selection_n = self.population_size - 1
            warnings.warn(f'Set tournament_selection_n={self.tournament_selection_n} to avoid Julia crash '
                          f'(because tournament_selection_n must be less than population_size={self.population_size})')
        # Activate interpretable mode
        if self.interpretable_mode:
            self.activate_interpretable_mode()

    def activate_interpretable_mode(self):
        if self.gaussian_fit:
            raise NotImplementedError('Constraints do not seem to be respected with template')
        self.unary_operators = ['square', 'sqrt', "inv(x) = 1/x", ]
        self.binary_operators = ["+", "-", "*"]
        self.constraints = {'*': (2, 1), 'square': 2, 'sqrt': 2, 'inv': 2}
        self.complexity_of_variables = 2
        self.set_params(extra_sympy_mappings={'inv': lambda x: 1 / x})
        self.set_params(constraints={'*': (2, 1), 'square': 2, 'sqrt': 2, 'inv': 2})

    @classmethod
    def get_run_id(cls, params: dict) -> str:
        non_default_params = cls.get_non_default_params(params)
        if 'model_selection' in non_default_params:
            non_default_params.pop('model_selection')
        return get_run_id(non_default_params)

    def fit_with_run(self, X: ndarray, y: ndarray, validation_mask: Optional[ndarray] = None,
            variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
            y_units: Optional[ArrayLike[str]] = None, run: Run = Optional) -> "PySRRegressor":
        """Method that implement additional options compared to PySR:
            -add some potential preprocessing before the fit
            -fit with TensorBoard logging
            -update/correct small difference in the loss of the self.equations_ dataframe

        Parameters & Results
        ----------
        Same as the self.fit method"""
        # Try loading emulator from file
        if run.has_been_saved:
            try:
                emulator_from_file = self.from_file(run_directory=run.run_directory)
            except (RuntimeError, EmptyDataError) as e:
                log_info(f'Catch the following error: {e.__repr__()}')
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
            self.julia_state_stream_ = emulator_from_file.julia_state_stream_
            #  Activate interpretable mode (because `extra_sympy_mappings` must be redefined at runtime=
            if self.interpretable_mode:
                self.activate_interpretable_mode()
        else:
            log_info(f'Start fit with {self.non_default_params}')
            #  Fit with logging and compute its duration
            start_time = time.monotonic()
            # By default, we log with tensorboard the progress for each iteration of the run
            # See https://github.com/MilesCranmer/PySR/discussions/840 for more details on log_interval
            logging = (self.logger_spec is True)
            if logging:
                self.logger_spec = run.get_logger_spec(log_interval=1 * self.populations)
            #  Fit
            self._fit(X, y, validation_mask, variable_names, X_units, y_units)
            if logging:
                self.logger_spec = True
            end_time = time.monotonic()
            duration = str(timedelta(seconds=end_time - start_time))
            # Save duration and tensorboard command to file
            log_info(f'Save fit to file')
            run.save_fit(duration, verbose=False)
            log_info(f'End fit with {self.non_default_params}')

        # Post-processing for Gaussian fit,
        if self.gaussian_fit:
            # Replace julia square by python power
            self.equations_['equation'] = self.equations_['equation'].apply(replace_julia_square_by_python_power)
            # Extract mu and sigma functions
            mu_functions, sigma_functions = [], []
            for equation_str in self.equations_['equation']:
                mu_equation_str, sigma_equation_str = [s.split('=')[-1] for s in equation_str.split(';')]
                mu_function = get_lambda_function_list(mu_equation_str, self.X_variable_names_for_gaussian_fit)
                mu_functions.append(mu_function)
                sigma_function = get_lambda_function_list(sigma_equation_str, self.X_variable_names_for_gaussian_fit, add_exponential=True)
                sigma_functions.append(sigma_function)
            # Add two columns
            self.equations_['mu'] = mu_functions
            self.equations_['sigma'] = sigma_functions
            # Add 'sympy_format' column
            def get_expr_function(sub_equation_number: int):
                def get_expr(equations_separated_by_semi_colon: str):
                    sub_equation = equations_separated_by_semi_colon.split(';')[sub_equation_number].split('=')[1]
                    syms = symbols(' '.join(variable_names)) if len(variable_names) > 1 else symbols(variable_names)
                    return sympify(sub_equation, locals={name: syms[i] for i, name in enumerate(variable_names)})
                return get_expr
            for sub_equation_number, expression_column_name in enumerate(self.expression_column_names):
                self.equations_[expression_column_name] = self.equations_['equation'].apply(get_expr_function(sub_equation_number))
        else:
            # Add 'equation' column
            self.equations_['equation'] = self.equations_['sympy_format'].apply(get_equation_from_expr)

        #  Some checks for the interpretable_mode
        if self.interpretable_mode:
            for expression_column_name in self.expression_column_names:
                for expr in self.equations_[expression_column_name]:
                    error_message = (f'{expr} is not interpretable,\n'
                                     f'this expression was obtained with {self.non_default_params}')
                    if not is_interpretable(expr):
                        pass
                    assert is_interpretable(expr), error_message

        # Some postprocessing on the 'equation' column
        self.equations_['equation'] = self.equations_['equation'].apply(postprocessing_for_equation)
        if self.interpretable_mode:
            self.equations_['equation'] = self.equations_['equation'].apply(lambda s: s.replace('*', ''))

        # Update 'loss' column if needed
        if self.metric_ is Metric.RMSE:
            #  Take the sqrt of the mean squared error
            self.equations_['loss'] = self.equations_['loss'].apply(np.sqrt)

        #  Add a 'validation_loss' column in self.equations_
        if validation_mask is not None:
            X_validation, y_validation = get_X_and_y(X, y, validation_mask, validation_set=True)
            self.equations_['validation_loss'] = self.compute_loss_list(X_validation, y_validation)
            #  Set the index for the 'validated' model selection using the validation set
            self.index_for_validated_model_selection_ = np.nanargmin(self.validation_loss_list)

        return self

    def _fit(self, X: ndarray, y: ndarray, validation_mask: Optional[ndarray] = None,
             variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
             y_units: Optional[ArrayLike[str]] = None):
        #  Extract X_fit and y_fit
        if validation_mask is None:
            X_fit, y_fit = X, y
        else:
            X_fit, y_fit = get_X_and_y(X, y, validation_mask, validation_set=False)
        # Modify X_fit and y_fit for Gaussian fit
        if self.gaussian_fit:
            if (X_units is not None) or (y_units is not None):
                # warn if some units were specified
                warnings.warn('Units are not accounted for in a gaussian fit, '
                              'because TemplateExpr does not handle units,'
                              'see https://github.com/MilesCranmer/PySR/discussions/869 ')
            X_units, y_units = None, None
            assert variable_names == self.X_variable_names_for_gaussian_fit
            X_fit = get_X_for_gaussian_fit(X_fit, y_fit)
            y_fit = np.zeros(len(X_fit))
            variable_names = self.X_variable_names_for_gaussian_fit + [self.y_variable_name_for_gaussian_fit]
        super().fit(X_fit, y_fit, variable_names=variable_names, X_units=X_units, y_units=y_units)
        #  Save checkpoint without the 2 columns containing julia objects, including dynamical equations
        if self.gaussian_fit and (not self.temp_equation_file):
            self.equations_.drop(columns=['julia_expression', 'lambda_format'], inplace=True)
            self.equations_['equation'] = self.equations_['equation'].apply(self.improve_equation_str)
            self._checkpoint()

    def improve_equation_str(self, equation: str) -> str:
        # renaming variable names from #1 -> x1 because it seems to hurt loading from pickle files
        equation = equation.replace('#', 'x')
        # Rename the function in the equation
        equation = equation.replace('mu', '\mu')
        equation = equation.replace('log_sigma', 'log(\sigma)')
        # Improve variable names in the equation (start with longer variable names to avoid bugs)
        for i, variable_name in list(enumerate(self.X_variable_names_for_gaussian_fit, 1))[::-1]:
            old_name, new_name = f'x{i}', variable_name
            equation = equation.replace(old_name, new_name)
        return equation

    def predict(self, X, index: int | list[int] | None = None, *, category: ndarray | None = None) -> ndarray:
        if self.metric_ is Metric.NLL:
            assert category is None
            return self.get_distri_param(X, 'mu', index)
        else:
            return super().predict(X, index, category=category)

    def predict_uncertainty_interval(self, X, index: int | list[int] | None = None,
                                     uncertainty_interval=UncertaintyInterval.plus_and_minus_std) -> ndarray:
        """Compute uncertainty intervals, i.e. a 2D array with the same length as X and with 2 columns
        The 1st column correspond to the lower error (it is negative) and the 2nd to the upper error.
        Note that uncertainty interval are only available for certain fit configurations."""
        if self.metric_ is Metric.NLL:
            sigma_values = self.get_distri_param(X, 'sigma', index)
            if uncertainty_interval is UncertaintyInterval.plus_and_minus_std:
                return np.array([(- sigma, sigma) for sigma in sigma_values])
            elif uncertainty_interval is UncertaintyInterval.ninety_percent:
                quantiles = [0.05, 0.95]
                rvs = [norm(loc=0, scale=sigma) for sigma in sigma_values]
                return np.array([[rv.ppf(quantile) for quantile in quantiles] for rv in rvs])
            else:
                raise NotImplementedError(uncertainty_interval)
        else:
            raise ValueError('uncertainty is not handled by the other metrics')


    def get_distri_param(self, X, distri_param_name: str, index: int | list[int] | None = None) -> ndarray:
        assert self.gaussian_fit
        assert distri_param_name in self.equations_.columns
        row = self.get_best() if index is None else self.equations_.iloc[index]
        return np.apply_along_axis(row[distri_param_name], axis=1, arr=X)

    """Method to compute the loss"""

    def compute_loss(self, X: ndarray, y: ndarray, index: int | list[int] | None = None) -> float:
        """Compute loss for the equation at some specific index"""
        if self.metric_ is Metric.NLL:
            row = self.get_best() if index is None else self.equations_.iloc[index]
            mu, sigma = [np.array([row[k](x) for x in X]) for k in ['mu', 'sigma']]
            return compute_loss_gaussian_fit(y, mu, sigma)
        elif self.metric_ is Metric.RMSE:
            return compute_loss(y, self.predict(X, index), self.metric_)
        else:
            raise NotImplementedError

    """Properties/method for the selected equations"""

    def compute_selected_loss(self, X: ndarray, y: ndarray) -> float:
        """Compute loss for the selected equation"""
        return self.compute_loss(X, y, index=None)

    def get_best(self, index: int | list[int] | None = None) -> pd.Series | list[pd.Series]:
        """Compute a Series (or list of Series) representing the selected equations (complexity, loss, ...)
         If index=None, then the equation is selected using self.model_selection"""
        if index is None:
            assert self.model_selection in ['best', 'validated', 'multiply']
            if self.model_selection == 'validated':
                assert self.index_for_validated_model_selection_ is not None
                index = self.index_for_validated_model_selection_
            elif self.model_selection == 'best':
                if self.metric_ is Metric.RMSE:
                    threshold = np.sqrt(1.5) * self.equations_["loss"].min()
                elif self.metric_ is Metric.NLL:
                    threshold = 1.5  * self.equations_["loss"].min()
                else:
                    raise NotImplementedError
                filtered_equations = self.equations_.query(f"loss <= {threshold}")
                index = filtered_equations["score"].idxmax()
            else:
                raise NotImplementedError
        return self.equations_.iloc[index]

    @property
    def selected_row(self) -> pd.Series:
        """Selected row/pd.Series from the Dataframe self.equations_"""
        return self.get_best()

    @property
    def selected_complexity(self) -> int:
        """Complexity for the selected equation"""
        return self.selected_row['complexity']

    @property
    def selected_loss_train(self):
        return self.selected_row['loss']

    @property
    def selected_loss_validation(self) -> float:
        return self.selected_row['validation_loss']

    @property
    def selected_expressions(self) -> list[Expr]:
        """Sympy expressions for the selected equation,
        It returns a list of Expr because with 'gaussian_fit=True' we have two Expr in the selected equation"""
        return [self.selected_row[expression_column_name] for expression_column_name in self.expression_column_names]

    @property
    def expression_column_names(self) -> list[str]:
        return ['sympy_format_mu', 'sympy_format_sigma'] if self.gaussian_fit else ['sympy_format' ]

    @property
    def selected_equation(self) -> str:
        """Equation as a string"""
        return self.selected_row['equation']

    @property
    def selected_variable_names(self) -> list[str]:
        """List of variables names in the selected expression(s)
        In the case where there is two selected expressions (mu and sigma) we still return a single list of str"""
        list_of_variable_names = [get_variable_names(expr) for expr in self.selected_expressions]
        return list(chain.from_iterable(list_of_variable_names))

    @property
    def selected_variable_signed_names(self) -> list[str]:
        """List of variables signed names in the selected expression(s)
        In the case where there is two selected expressions (mu and sigma) we still return a single list of str"""
        list_of_variable_names = [get_variable_signed_names(expr) for expr in self.selected_expressions]
        return list(chain.from_iterable(list_of_variable_names))


    """Properties/method for every equation of the Pareto optimal set of equations"""

    def compute_loss_list(self, X: ndarray, y: ndarray) -> list[float]:
        """Compute a list of loss: one loss for every equation of the Pareto optimal set of equations"""
        return [self.compute_loss(X, y, index) for index in range(len(self.equations_))]

    @property
    def complexity_list(self) -> list[int]:
        """List of complexity for the equations of the Pareto front (thus in increasing order)"""
        return self.equations_['complexity'].to_list()

    @property
    def loss_list(self) -> list[float]:
        """List of Train loss for the equations of the Pareto front"""
        return self.equations_['loss'].to_list()

    @property
    def validation_loss_list(self) -> list[float]:
        """List of Validation loss for the equations of the Pareto front"""
        return self.equations_['validation_loss'].to_list()

    @property
    def equation_list(self) -> list[str]:
        """List of equations as string that belongs to the Pareto front"""
        return self.equations_['equation'].to_list()

    """Other methods/properties"""

    @property
    def metric_(self) -> Metric:
        return Metric.NLL if self.gaussian_fit else Metric.RMSE

    @property
    def params(self):
        return self.get_params()

    @property
    def non_default_params(self) -> dict[str, Any]:
        return self.get_non_default_params(self.params)

    @classmethod
    def get_non_default_params(cls, params):
        return get_non_default_params(params, cls)

    def __repr__(self) -> str:
        """If we do not override this method, then the __repr__ method from PySR fails
        because it does not handle model_selection='validated'"""
        pass

    def remove_folder(self):
        self.run_.remove_folder()