import os.path as op
from typing import Literal, Callable, Optional

import numpy as np
import pandas as pd
from numpy import ndarray
from pysr import AbstractExpressionSpec, AbstractLoggerSpec, PySRRegressor
from pysr.utils import ArrayLike
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection._search import BaseSearchCV

from data.utils_dataset.utils_validation import compute_default_validation_mask
from data.utils_search.search_experiment import SearchExperiment
from data.utils_search.utils_non_default_params import get_non_default_params
from data.utils_search.utils_search_path import get_search_path
from emulator.pysr_emulator import PySREmulator
from emulator.utils_metric.metric import Metric, metric_to_function
from emulator_with_search.utils_attributes.utils_search_cv import get_search_cv_kwargs
from emulator_with_search.utils_attributes.utils_validation import get_cv, get_X_and_y
from emulator_with_search.utils_cv_results.utils_df_results import get_df_cv_results, METRIC_COLUMN_NAME, \
    RMSE_VALIDATION_COLUMN_NAME
from emulator_with_search.utils_cv_results.utils_optimize_threshold import compute_optimal_threshold
from emulator_with_search.utils_param_grid.utils_scaling_factor import get_param_grid
from emulator_with_search.utils_param_grid.utils_search_style import search_style_to_search_cv_type
from utils.utils_log import log_info


class PySREmulatorWithSearch(PySREmulator):
    """This class is an extension of PySREmulator with hyperparameter search. Hyperparameter settings are
    compared on a validation set, and the best hyperparameter setting (minimizing validation error) is selected

    This extension has several additional attributes:

        validation_size: float
            represent the proportion (between 0 and 1) of data to include in the validation split.
            Default is 0.3
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
            Default is 10
    """
    validation_mask_: Optional[np.ndarray[bool]]
    search_experiment_: Optional[SearchExperiment]

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
        self.search_style = 'random' if search_style is None else search_style
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.param_grid = dict() if param_grid is None else param_grid
        self.param_list_to_optimize = param_list_to_optimize
            # Hyperparameters that could be added: 'populations', 'population_size' (but can lead to long computation)
        self.scaling_factor = scaling_factor
        # Some checks
        assert isinstance(self.validation_size, float) and (0 < self.validation_size < 1)
        assert isinstance(self.search_style, str)
        assert isinstance(self.n_iter, int) and self.n_iter > 0
        assert isinstance(self.param_grid, (dict, list))
        assert (self.n_jobs is None) or isinstance(self.n_jobs, int)
        #  Set param grid using param_list_to_optimize if param_grid has not been specified by the user
        if not self.param_grid:
            self.param_grid = get_param_grid(self, self.scaling_factor, self.search_style, self.n_iter, 
                                             self.param_list_to_optimize)
        # Create attributes
        self.validation_mask_ = None
        self.search_experiment_ = None

    def fit(self, X, y, *, Xresampled=None, weights=None, variable_names: ArrayLike[str] | None = None,
            complexity_of_variables: int | float | list[int | float] | None = None,
            X_units: ArrayLike[str] | None = None, y_units: str | ArrayLike[str] | None = None,
            category: ndarray | None = None, validation_mask: np.ndarray[bool] = None) -> "PySRRegressor":
        """
        Fit where many hyperparameters settings are compared on a single validation set, and the hyperparameter
        setting that minimizes the validation error is selected
        Some arguments from the fit() method of PySR, are not yet handled (weights, Xresampled, ...)
        because we would need to modify search path for every variation of these arguments.
        We add one optional argument:
             validation_mask: array of boolean s.t. validation_mask[i] indicates if the index 'i' is in the validation set
        """
        # Set validation_mask
        self.validation_mask_ =  compute_default_validation_mask(y) if validation_mask is None else validation_mask
        # Run hyperparameter search experiment
        self.search_experiment_ = self.run_hyperparameter_search(X, y, variable_names=variable_names,
                                                                 X_units=X_units, y_units=y_units)
        # Fit with the best setting of hyperparameter on the train split
        return self.fit_with_best_params(X, y, variable_names=variable_names, X_units=X_units, y_units=y_units)

    def fit_with_best_params(self, X: np.ndarray, y: np.ndarray, **params_fit):
        #  By default, we log with tensorboard the progress of this fit iteration by iteration
        assert self.logger_spec is None
        self.set_params(**self.search_experiment_.best_params)
        self.logger_spec = self.search_experiment_.get_logger_spec(log_interval=1 * self.populations)
        X_train_train, y_train_train = get_X_and_y(X, y, self.validation_mask_, validation_set=False)
        super().fit(X_train_train, y_train_train, **params_fit)
        self.logger_spec = None
        return self

    def run_hyperparameter_search(self, X: np.ndarray, y: np.ndarray, **params_fit) -> SearchExperiment:
        """Run hyperparameter search, save search results to file, and return search_experiment"""
        log_info(f'Run hyperparameter search with param grid = {self.param_grid}')
        non_default_params = get_non_default_params(self)
        search_experiment = SearchExperiment(get_search_path(X, y, self.validation_mask_, non_default_params))
        # Compute and save search results only it has not yet been saved
        if not op.exists(search_experiment.filepath_search_result):
            search_experiment.save_search_results(self.compute_df_cv_results(X, y, **params_fit), non_default_params)
        log_info(f'Best results from the hyperparameter search:\n{search_experiment}')
        return search_experiment

    def compute_df_cv_results(self, X: np.ndarray, y: np.ndarray, **params_fit) -> pd.DataFrame:
        """Run hyperparameter search and return the results transformed as a DataFrame called df_cv_results"""
        log_info('Compute search results')

        # Run hyperparameter search with respect to self.param_grid
        log_info('Start hyperparameter search')
        search_cv_type = search_style_to_search_cv_type[self.search_style]
        search_cv = self.run_search_cv(search_cv_type, X, y, self.param_grid, **params_fit)

        # Compute optimal threshold for each emulator
        log_info('Compute optimal threshold')
        X_validation, y_validation = get_X_and_y(X, y, self.validation_mask_, validation_set=True)
        emulators = search_cv.cv_results_['estimator']
        for j, emulator in enumerate(emulators):
            assert isinstance(emulator, PySREmulator)
            assert emulator.equations_ is not None
            optimal_threshold = compute_optimal_threshold(emulator, X_validation, y_validation)
            emulator.threshold_for_model_selection = optimal_threshold
            search_cv.cv_results_['params'][j]['threshold_for_model_selection'] = optimal_threshold
        # Add RMSE validation information
        rmse_validation_list = [emulator.compute_loss(X_validation, y_validation, Metric.RMSE) for emulator in emulators]
        search_cv.cv_results_[RMSE_VALIDATION_COLUMN_NAME] = rmse_validation_list

        #  Transform cv_results into a Dataframe sorted by ranking with additional columns
        return get_df_cv_results(search_cv.cv_results_)

    def run_search_cv(self, search_cv_type: type, X, y, param_grid: dict | list[dict], **params_fit):
        """Run hyperparameter search for a specific type of search (random, grid), a param_grid (all hyperparameters)
        and some parameters 'params_fit' that will be passed to the estimator"""
        assert issubclass(search_cv_type, BaseSearchCV)
        search_cv = search_cv_type(estimator=self.load_pysr_emulator_with_same_attributes(),
                                   scoring={'MSE': make_scorer(mean_squared_error, greater_is_better=False)},
                                   cv=get_cv(self.validation_mask_), refit=False, return_train_score=False,
                                   n_jobs=self.n_jobs,
                                   **get_search_cv_kwargs(search_cv_type, param_grid, self.n_iter))
        search_cv.fit(X, y, **params_fit)
        return search_cv

    def load_pysr_emulator_with_same_attributes(self) -> PySREmulator:
        """Load a pysr_emulator object with the same attributes as self,
        except additional attributes that are due to inheritance"""
        estimator = PySREmulator()
        params = self.get_params()
        params = {param_name: params[param_name] for param_name in estimator.__dict__ if param_name in params}
        estimator.set_params(**params)
        return estimator



