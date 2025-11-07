import pytest
from sympy import symbols, exp, log, sqrt

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from data.utils_run.run import Run
from emulator.emulator import Emulator
from emulator.is_interpretable import is_interpretable


def test_interpretable():
    x, y = symbols('x y')
    expr = 1 + 2 * x + 3 * y + 4 * (x ** 2) + 5 / y + 6 / (y ** 2) + 7 * sqrt(x) + 8 / sqrt(y)
    assert is_interpretable(expr)
    expr = x * y
    assert not is_interpretable(expr)
    expr = x ** 3
    assert not is_interpretable(expr)
    expr = 1 / y ** 3
    assert not is_interpretable(expr)
    expr = x / y
    assert not is_interpretable(expr)
    expr = exp(x)
    assert not is_interpretable(expr)
    expr = log(x)
    assert not is_interpretable(expr)


def test_corner_case():
    dataset = get_dataset(validation_size=0.3, validation_split=ValidationSplit.QUANTILE_WITH_BINNING)
    params = {'adaptive_parsimony_scaling': 892.3991044718146, 'crossover_probability': 0.22308731969125997, 'fraction_replaced': 0.002460616067526847, 'fraction_replaced_hof': 0.2799804753022064, 'interpretable_mode': True, 'logger_spec': None, 'maxdepth': 2, 'maxsize': 38, 'ncycles_per_iteration': 3243, 'niterations': 413, 'optimize_probability': 0.3741523922560336, 'optimizer_f_calls_limit': 1567, 'output_directory': '/home/e23lerou/Documents/EquationDiscovery4ClimateImpact/data/run/ed63224f4d5f8a7230b7434ae0662453', 'perturbation_factor': 0.30133929183877456, 'population_size': 28, 'populations': 3, 'probability_negate_constant': 0.037166727977054705, 'run_id': 'b7079bd524c88c28d5778c7eb3ea40e0', 'timeout_in_seconds': 3600, 'topn': 2, 'tournament_selection_n': 16, 'tournament_selection_p': 0.3329019834400152, 'warmup_maxsize_by': 0.6962700559185838, 'weight_add_node': 1.0378129974300148, 'weight_delete_node': 0.9542348970550681, 'weight_do_nothing': 0.33851912194589834, 'weight_insert_node': 0.02161666202867044, 'weight_mutate_constant': 0.33558151839447187, 'weight_mutate_operator': 1.040239323431594, 'weight_optimize': 0.9455490474077702, 'weight_randomize': 0.004497312966155328, 'weight_rotate_tree': 6.686664861569644, 'weight_simplify': 0.01928349979686331, 'weight_swap_operands': 0.029761183277212507}
    emulator = Emulator(**params)
    emulator.fit(dataset.X_train, dataset.y_train, dataset.validation_mask,
             dataset.X_variable_names, dataset.X_units, dataset.y_units)
    emulator.remove_folder()







