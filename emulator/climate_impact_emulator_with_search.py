import math
import os.path as op
from typing import Literal, Callable, Optional

import numpy as np
import pandas as pd
from pysr import AbstractExpressionSpec, AbstractLoggerSpec, PySRRegressor
from pysr.utils import ArrayLike
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection._search import BaseSearchCV, GridSearchCV

from data.search.utils_json_loader import string_to_dict
from data.search.utils_search import get_filepath_search
from emulator.climate_impact_emulator import ClimateImpactEmulator
from emulator.utils_hyperparameter_search.utils_search_cv import get_search_cv_kwargs
from emulator.utils_hyperparameter_search.utils_validation import compute_ind_validation, get_cv, get_X_and_y
from emulator.utils_optimize_threshold.utils_threshold import get_param_grid_with_thresholds
from utils.utils_log import log_info


class ClimateImpactEmulatorWithSearch(ClimateImpactEmulator):
    """This class is an extension of ClimateImpactEmulator with hyperparameter search. Hyperparameter settings are
    compared on a validation set, and the best hyperparameter setting (minimizing validation error) is selected

    This extension has several additional attributes:

        validation_size: float
            represent the proportion (between 0 and 1) of data to include in the validation split.
            Default is 0.3
        search_cv_type: type
            Class for hyperparameter search with a single validation
            Default is RandomSearchCV
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
        param_list_to_optimize_around_default: list[str]
            List of hyperparameter names that are optimized by random search around default
            If param_grid is specified, i.e. different from None, then this list is not accounted for
            Default is None, this default is replaced by a list of 3 defaults hyperparameters that are optimized
        scaling_factor: int
            Scaling factor to optimize around default.
            Hyperparameter are sampled in [default_value / scaling_factor, default * scaling_factor]
            Default is 10
        save_or_load_csv_of_search_results: bool
            Whether search results should be saved to a csv (or loaded from a csv if the search has been run)
            Default is True

    """
    df_ranked_results_: Optional[pd.DataFrame]
    ind_validation_: Optional[np.ndarray[bool]]

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
                 select_k_features: int | None = None, threshold_for_best_model_selection: float = 1.5,
                 feature_selection_name: str = 'PySRDefault',
                 # Additional parameters
                 validation_size: float = 0.3,
                 search_cv_type: type = RandomizedSearchCV,
                 n_iter: int = 10,
                 n_jobs: Optional[int] = None,
                 param_grid: dict[str, list] | list[dict[str, list]] = None,
                 param_list_to_optimize_around_default: Optional[list[str]] = None,
                 scaling_factor: int = 10,
                 save_or_load_csv_of_search_results: bool = True,
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
                         threshold_for_best_model_selection=threshold_for_best_model_selection,
                         feature_selection_name=feature_selection_name, **kwargs)
        self.validation_size = validation_size
        self.search_cv_type = search_cv_type
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.param_grid = dict() if param_grid is None else param_grid
        self.param_list_to_optimize_around_default = param_list_to_optimize_around_default
        if self.param_list_to_optimize_around_default is None:
            self.param_list_to_optimize_around_default = ['niterations', 'adaptive_parsimony_scaling', 'fraction_replaced_hof']
            # Hyperparameters that could be added: 'populations', 'population_size' (but can lead to long computation)
        self.scaling_factor = scaling_factor
        self.save_or_load_csv_of_search_results = save_or_load_csv_of_search_results
        # Some checks
        assert isinstance(self.validation_size, float) and (0 < self.validation_size < 1)
        assert issubclass(self.search_cv_type, BaseSearchCV)
        assert isinstance(self.n_iter, int) and self.n_iter > 0
        assert (self.n_jobs is None) or isinstance(self.n_jobs, int)
        assert isinstance(self.param_grid, (dict, list))
        assert isinstance(self.param_list_to_optimize_around_default, list)
        assert isinstance(save_or_load_csv_of_search_results, bool)
        # Set param grid with param_list_to_optimize_around_default if it has not been specified by the user
        # Hyperparameters in the list should be sampled between [default_value / scaling_factor, default * scaling_factor]
        if not self.param_grid:
            for key in self.param_list_to_optimize_around_default:
                default_value = self.__getattribute__(key)
                min_value = default_value / self.scaling_factor
                max_value = default_value * self.scaling_factor
                if isinstance(default_value, int):
                    min_value = math.ceil(min_value)
                self.param_grid[key] = [min_value, max_value]
        # Some checks
        if 'population_size' in self.param_grid:
            min_population_size = min(self.param_grid['population_size'])
            assert self.tournament_selection_n < min_population_size, \
                (f"tournament_selection_n parameter (={self.tournament_selection_n}) "
                 f"must be smaller than the minimum population_size (={min_population_size})")
        # Create attributes
        self.df_ranked_results_ = None
        self.ind_validation_ = None

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, variable_names: ArrayLike[str] | None = None,
            X_units: ArrayLike[str] | None = None, y_units: str | ArrayLike[str] | None = None,
            index_start_validation: int = 0) -> "PySRRegressor":
        """
        Fit where many hyperparameters settings are compared on a single validation set, and the hyperparameter
        setting that minimizes the validation error is selected
        Some arguments from the fit() method of PySR, are not yet handled (weights, Xresampled, ...)
        because we would need to modify filepath_search for every variation of these arguments.
        We add one argument:
             index_start_validation: int; first index for the validation; Default is 0
        """
        # Some standard checks
        assert X.shape[0] == y.shape[0]
        assert isinstance(index_start_validation, int)
        # Compute an array of boolean such that ind_validation[i] = True if the index 'i' is in the validation set
        self.ind_validation_ = compute_ind_validation(len(y), self.validation_size, index_start_validation)
        # Compute the attribute df_ranked_results_, a Dataframe with the result of the hyperparameter search
        self.df_ranked_results_ = self.get_df_ranked_results(X, y, variable_names=variable_names,
                                                             X_units=X_units, y_units=y_units,
                                                             use_cache=True)
        # Fit with the best setting of hyperparameter (best_params) on the train split
        best_params = self.df_ranked_results_.iloc[0].loc['params']
        self.set_params(**best_params)
        X_train_train, y_train_train = get_X_and_y(X, y, self.ind_validation_, validation_set=False)
        return super().fit(X_train_train, y_train_train, variable_names=variable_names, X_units=X_units,
                           y_units=y_units, use_cache=False)

    def get_df_ranked_results(self, X, y, **params_fit) -> pd.DataFrame:
        """Load or run hyperparameter search to obtain df_ranked_results"""
        X_sum, y_sum = self.get_X_sum_and_y_sum(X, y)
        filepath_search = get_filepath_search(X_sum, y_sum, self.validation_size, self.search_cv_type, self.n_iter,
                                              self.param_grid, self.feature_selection_name, self.select_k_features, **params_fit)
        if op.exists(filepath_search) and self.save_or_load_csv_of_search_results:
            log_info('Load df_ranked_results from csv file')
            df_ranked_results = pd.read_csv(filepath_search, index_col=0)
            df_ranked_results['params'] = df_ranked_results['params'].apply(string_to_dict)
        else:
            log_info('Compute df_ranked_results')
            df_ranked_results = self.compute_df_ranked_results(X, y, **params_fit)
            if self.save_or_load_csv_of_search_results:
                log_info('Save df_ranked_results to csv file')
                df_ranked_results.to_csv(filepath_search)
        return df_ranked_results

    def compute_df_ranked_results(self, X, y, **params_fit) -> pd.DataFrame:
        """Run 2 consecutive hyperparameter search (first self.search_cv_type, then a grid search for thresholds)
        and save the ranked results in the attribute df_ranked_results_"""
        #  Run hyperparameter search with respect to self.param_grid
        log_info('Start first hyperparameter search')
        search_cv = self.run_search_cv(self.search_cv_type, X, y, self.param_grid, **params_fit)
        # Run a grid search that extends the first hyperparameter search with a list of thresholds to try.
        # This additional grid search for the threshold cost almost nothing because fit results have been cached
        log_info('Start second hyperparameter search for threshold')
        search_cv = self.run_search_cv(GridSearchCV, X, y, get_param_grid_with_thresholds(search_cv), **params_fit)
        #  Extract best params from search cv results
        df_results = pd.DataFrame(search_cv.cv_results_)
        column_for_ranking = 'rank_test_MSE'
        df_ranked_results_ = df_results.sort_values(by=column_for_ranking)
        assert df_ranked_results_[column_for_ranking].values[0] == 1
        return df_ranked_results_

    def run_search_cv(self, search_cv_type: type, X, y, param_grid: dict, **params_fit):
        """Run hyperparameter search for a specific type of search (random, grid), a param_grid (all hyperparameters)
        and some parameters 'params_fit' that will be passed to the estimator"""
        assert issubclass(search_cv_type, BaseSearchCV)
        search_cv = search_cv_type(estimator=self.load_climate_impact_emulator_with_same_attributes(),
                                   scoring={'MSE': make_scorer(mean_squared_error, greater_is_better=False)},
                                   cv=get_cv(self.ind_validation_), n_jobs=self.n_jobs, refit=False,
                                   return_train_score=True,
                                   error_score='raise',
                                   **get_search_cv_kwargs(search_cv_type, param_grid, self.n_iter))
        search_cv.fit(X, y, **params_fit)
        return search_cv

    def load_climate_impact_emulator_with_same_attributes(self) -> ClimateImpactEmulator:
        """Load a climate_impact_emulator object with the same attributes as self,
        except additional attributes that are due to inheritance"""
        estimator = ClimateImpactEmulator()
        params = self.get_params()
        params = {param_name: params[param_name] for param_name in estimator.__dict__}
        estimator.set_params(**params)
        return estimator



