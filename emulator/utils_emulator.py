import numpy as np

params_that_do_not_impact_the_fit_results = {'logger_spec', 'output_directory', 'run_id',
                                             'parallelism', 'procs', 'cluster_manager',
                                             'deterministic', 'verbosity', 'update_verbosity', 'progress',
                                             'input_stream', 'temp_equation_file', 'tempdir', 'delete_tempfiles', 'extra_sympy_mappings',
                                             'extra_torch_mappings', 'extra_jax_mappings', 'update', 'n_jobs',

                                                # The following params impact the results, but they both directly
                                             # depend on 'gaussian_fit' params, so  we do not need to include
                                             'expression_spec', 'elementwise_loss', 'loss_function'}

class Config:
    automatic_loading_and_saving = True

def get_X_for_gaussian_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.concat([X, np.expand_dims(y, axis=1)], axis=1)
