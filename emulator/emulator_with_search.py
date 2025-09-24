import os.path as op
from math import prod
from typing import Literal, Callable, Optional

import numpy as np
from pysr import AbstractExpressionSpec, AbstractLoggerSpec, PySRRegressor
from pysr.utils import ArrayLike
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection._search import BaseSearchCV

from data.utils_run.run import Run
from data.utils_run.utils_run import get_run_id
from emulator.emulator import Emulator
from emulator.utils_hyperparameter_search.utils_column_names import PARAMS_EMULATOR_COLUMN_NAME
from emulator.utils_hyperparameter_search.utils_df_cv_results import compute_df_cv_results
from emulator.utils_hyperparameter_search.utils_scaling_factor import get_param_grid
from emulator.utils_hyperparameter_search.utils_search_cv import get_search_cv_kwargs, get_cv
from emulator.utils_hyperparameter_search.utils_search_style import search_style_to_search_cv_type
from utils.utils_log import log_info
from utils.utils_non_default_params import get_non_default_params


class EmulatorWithSearch(Emulator):
    """EmulatorWithSearch is an extension of Emulator with hyperparameter search.
     Several hyperparameter settings are compared on the validation set,
     and the top hyperparameter setting (minimizing validation error) is selected for the final 'fit' of the emulator

    -> additional parameters:
        search_style: str
            Style for hyperparameter search with a single validation, Possibilities include 'random' and 'grid' 
            Default is None, which will be replaced by 'random'
        n_iter: int
            Number of parameter settings sampled for RandomSearchCV, which trades off runtime vs quality of the solution
            Default is 10
        n_jobs : int
            Number of jobs to run in parallel.
            None means 1 unless in a joblib context. -1 means using all processors
            Default is None
        param_grid : dict[str, list] | list[dict[str, list]]
            Dictionary with hyperparameters names (`str`) as keys and lists of hyperparameter settings to try as values,
            or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are explored.
            This enables searching over any sequence of hyperparameter settings.
            Default is None, this default is replaced by an empty dictionary in the __init__ method
        param_list_to_optimize: list[str]
            List of hyperparameter names that are optimized, i.e. specified inside the param_grid
            If param_grid is specified, i.e. different from None, then this list is not accounted for
            Default is None, which leads to optimizing only the hyperparameter "niterations"
        scaling_factor: int
            Scaling factor to optimize around default.
            Hyperparameter are sampled in [default_value / scaling_factor, default * scaling_factor]
            Default is 10"""

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
                 print_precision: int = 5, progress: bool = True, logger_spec: AbstractLoggerSpec | None = None,
                 input_stream: str = "stdin", run_id: str | None = None, output_directory: str | None = None,
                 temp_equation_file: bool = False, tempdir: str | None = None, delete_tempfiles: bool = True,
                 update: bool = False, output_jax_format: bool = False, output_torch_format: bool = False,
                 extra_sympy_mappings: dict[str, Callable] | None = None,
                 extra_torch_mappings: dict[Callable, Callable] | None = None,
                 extra_jax_mappings: dict[Callable, str] | None = None, denoise: bool = False,
                 select_k_features: int | None = None,
                 # Additional parameters
                 search_style: Optional[str] = None,
                 n_iter: int = 10,
                 n_jobs: Optional[int] = None,
                 param_grid: dict[str, list] | list[dict[str, list]] = None,
                 param_list_to_optimize: Optional[list[str]] = None,
                 scaling_factor: int = 10,
                 **kwargs):
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
        self.search_style = 'random' if search_style is None else search_style
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.param_grid = dict() if param_grid is None else param_grid
        self.param_list_to_optimize = param_list_to_optimize
            # Hyperparameters that could be added: 'populations', 'population_size' (but can lead to long computation)
        self.scaling_factor = scaling_factor
        # Some checks
        assert isinstance(self.search_style, str)
        assert isinstance(self.n_iter, int) and self.n_iter > 0

        assert (self.n_jobs is None) or isinstance(self.n_jobs, int)
        #  Set param grid using param_list_to_optimize if param_grid has not been specified by the user
        if not self.param_grid:
            self.param_grid = get_param_grid(self, self.scaling_factor, self.search_style, self.n_iter, 
                                             self.param_list_to_optimize)
        # Some checks on param_grid
        assert isinstance(self.param_grid, (dict, list))

    def fit(self, X: np.ndarray, y: np.ndarray, validation_mask: Optional[np.ndarray[bool]] = None,
            variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
            y_units: Optional[ArrayLike[str]] = None) -> "PySRRegressor":
        """Fit the emulator for some feature X, target y, and validation_mask.
        Additional information can be specified: variable_names & units (with X_units, y_units)
        Run hyperparameter search with several hyperparameter settings (load from file the results if it exists)
        The top hyperparameter setting (minimizing validation error) is selected for the final 'fit' of the emulator

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

        # Run hyperparameter search
        if not op.exists(self.run_.filepath_search_result):
            self.run_and_save_hyperparameter_search(X, y, validation_mask, variable_names, X_units, y_units)

        # Fit with the top setting of hyperparameter on the train split
        log_info("Fit with top params")
        top_params_emulator = self.run_.top_params_emulator
        self.set_params(**top_params_emulator)
        run = Run(self.output_directory, get_run_id(get_non_default_params(top_params_emulator, Emulator)))
        assert run.has_been_saved, f'{run.run_directory}'

        # Fit with a specific run
        return self.fit_with_run(X, y, validation_mask, variable_names, X_units, y_units, run)

    @property
    def nb_combinations(self) -> int:
        if self.search_style == 'random':
            return self.n_iter
        elif self.search_style == 'grid':
            return prod([len(grid) for grid in self.param_grid.values()])
        else:
            raise NotImplementedError

    def run_and_save_hyperparameter_search(self, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool],
                                           variable_names: ArrayLike[str] | None = None, X_units: ArrayLike[str] | None = None,
                                           y_units: str | ArrayLike[str] | None = None) -> None:
        """Run hyperparameter search and save the results as a csv"""
        log_info(f'Start hyperparameter search with {self.nb_combinations} combinations, with param grid = {self.param_grid}')
        # Run a simple/fast regressor fit, just to load julia before using multiprocessing
        if self.n_jobs is not None:
            PySRRegressor(niterations=1, verbosity=0).fit(X, y)
        # Some check
        assert validation_mask is not None
        # Run hyperparameter search with respect to self.param_grid
        search_cv_type = search_style_to_search_cv_type[self.search_style]
        assert issubclass(search_cv_type, BaseSearchCV)
        search_cv = search_cv_type(estimator=self.load_emulator_with_same_attributes(),
                                   scoring={'MSE': make_scorer(mean_squared_error, greater_is_better=False)},
                                   cv=get_cv(validation_mask), refit=False, return_train_score=False,
                                   n_jobs=self.n_jobs,
                                   **get_search_cv_kwargs(search_cv_type, self.param_grid, self.n_iter))
        search_cv.fit(X, y, validation_mask=validation_mask, variable_names=variable_names,
                      X_units=X_units, y_units=y_units)
        # Transform cv_results into a Dataframe sorted by ranking with additional columns
        df_cv_results = compute_df_cv_results(search_cv.cv_results_, X, y, validation_mask)
        # Save df_cv_results to file
        self.run_.save_search_results(df_cv_results, self.non_default_params)

    def load_emulator_with_same_attributes(self) -> Emulator:
        """Load an emulator object with the same attributes as self,
        except additional attributes that are due to inheritance"""
        emulator = Emulator()
        params = self.get_params()
        params = {param_name: params[param_name] for param_name in emulator.__dict__ if param_name in params}
        emulator.set_params(**params)
        return emulator

    def remove_folder(self):
        for params in self.run_.df_cv_results[PARAMS_EMULATOR_COLUMN_NAME].values:
            Run(self.output_directory, get_run_id(get_non_default_params(params, Emulator))).remove_folder()
        super().remove_folder()


