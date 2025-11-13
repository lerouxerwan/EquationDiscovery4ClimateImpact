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


def run_corner_case(validation_size, validation_split, params):
    dataset = get_dataset(validation_size=validation_size, validation_split=validation_split)
    emulator = Emulator(**params)
    emulator.fit(dataset.X_train, dataset.y_train, dataset.validation_mask,
             dataset.X_variable_names, dataset.X_units, dataset.y_units)
    emulator.remove_folder()


def test_corner_case_1():
    params = {'adaptive_parsimony_scaling': 892.3991044718146, 'crossover_probability': 0.22308731969125997, 'fraction_replaced': 0.002460616067526847, 'fraction_replaced_hof': 0.2799804753022064, 'interpretable_mode': True, 'logger_spec': None, 'maxdepth': 2, 'maxsize': 38, 'ncycles_per_iteration': 3243, 'niterations': 413, 'optimize_probability': 0.3741523922560336, 'optimizer_f_calls_limit': 1567, 'output_directory': '/home/e23lerou/Documents/EquationDiscovery4ClimateImpact/data/run/ed63224f4d5f8a7230b7434ae0662453', 'perturbation_factor': 0.30133929183877456, 'population_size': 28, 'populations': 3, 'probability_negate_constant': 0.037166727977054705, 'run_id': 'b7079bd524c88c28d5778c7eb3ea40e0', 'timeout_in_seconds': 3600, 'topn': 2, 'tournament_selection_n': 16, 'tournament_selection_p': 0.3329019834400152, 'warmup_maxsize_by': 0.6962700559185838, 'weight_add_node': 1.0378129974300148, 'weight_delete_node': 0.9542348970550681, 'weight_do_nothing': 0.33851912194589834, 'weight_insert_node': 0.02161666202867044, 'weight_mutate_constant': 0.33558151839447187, 'weight_mutate_operator': 1.040239323431594, 'weight_optimize': 0.9455490474077702, 'weight_randomize': 0.004497312966155328, 'weight_rotate_tree': 6.686664861569644, 'weight_simplify': 0.01928349979686331, 'weight_swap_operands': 0.029761183277212507}
    run_corner_case(0.3, ValidationSplit.QUANTILE_WITH_BINNING, params)

@pytest.mark.xfail
def test_corner_case_2():
    params = {'adaptive_parsimony_scaling': 1749.494188956888, 'binary_operators': ['+', '-', '*'], 'complexity_of_variables': 2, 'constraints': {'*': (2, 1), 'square': 2, 'sqrt': 2, 'inv': 2}, 'crossover_probability': 0.2444040755561205, 'fraction_replaced': 0.0008938288249710478, 'fraction_replaced_hof': 0.07999582659827606, 'interpretable_mode': True, 'logger_spec': None, 'maxdepth': 3, 'maxsize': 36, 'ncycles_per_iteration': 743, 'niterations': 37, 'optimize_probability': 0.8343522450717555, 'optimizer_f_calls_limit': 52812, 'output_directory': '/home/e23lerou/Documents/EquationDiscovery4ClimateImpact/data/run/4ae879cf1871512b843f09e9402b98d8', 'perturbation_factor': 0.6362362177340899, 'population_size': 41, 'populations': 7, 'probability_negate_constant': 0.05628161915279, 'run_id': '6025267500d7330786cc54d91d9a3f4f', 'timeout_in_seconds': 3600, 'topn': 5, 'tournament_selection_n': 13, 'tournament_selection_p': 0.7556416094454519, 'unary_operators': ['square', 'sqrt', 'inv(x) = 1/x'], 'warmup_maxsize_by': 0.7988027018497522, 'weight_add_node': 5.070147904779255, 'weight_delete_node': 0.1969594358122144, 'weight_do_nothing': 0.33590501268579814, 'weight_insert_node': 0.11030018071847747, 'weight_mutate_constant': 0.3245529082979231, 'weight_mutate_operator': 0.03574498276070293, 'weight_optimize': 0.24833333415649342, 'weight_randomize': 0.0007048656583144816, 'weight_rotate_tree': 12.060614027361142, 'weight_simplify': 0.017129796260070048, 'weight_swap_operands': 0.052928297286970946}
    run_corner_case(0.3, ValidationSplit.RANDOM, params)

@pytest.mark.xfail
def test_corner_case_3():
    params = {'adaptive_parsimony_scaling': 5841.541767434696, 'binary_operators': ['+', '-', '*'], 'complexity_of_variables': 2, 'constraints': {'*': (2, 1), 'square': 2, 'sqrt': 2, 'inv': 2}, 'crossover_probability': 0.2432870162201439, 'fraction_replaced': 0.0029053527949135117, 'fraction_replaced_hof': 0.5231736787347895, 'interpretable_mode': True, 'logger_spec': None, 'maxdepth': 2, 'maxsize': 24, 'ncycles_per_iteration': 176, 'niterations': 17, 'optimize_probability': 0.19589612249450167, 'optimizer_f_calls_limit': 23807, 'output_directory': '/home/e23lerou/Documents/EquationDiscovery4ClimateImpact/data/run/5f991ee22b8064cfd79d2b58e5b3e9cc', 'perturbation_factor': 0.042675492580085767, 'population_size': 23, 'populations': 307, 'probability_negate_constant': 0.014566269683322771, 'run_id': 'e85343aa555efd9a6f75172b08d8ff6e', 'timeout_in_seconds': 3600, 'topn': 2, 'tournament_selection_n': 2, 'tournament_selection_p': 0.6218248106501165, 'unary_operators': ['square', 'sqrt', 'inv(x) = 1/x'], 'warmup_maxsize_by': 0.6650774434018913, 'weight_add_node': 1.327861581381353, 'weight_delete_node': 1.024108388899371, 'weight_do_nothing': 1.030295276903795, 'weight_insert_node': 0.09183139018351547, 'weight_mutate_constant': 0.33158320241563816, 'weight_mutate_operator': 1.3043783812418197, 'weight_optimize': 0.608357293045684, 'weight_randomize': 0.000943059188815957, 'weight_rotate_tree': 0.6744297424801324, 'weight_simplify': 0.0013369055141455876, 'weight_swap_operands': 0.4263014931858433}
    run_corner_case(0.25, ValidationSplit.MIDDLE, params)

@pytest.mark.xfail
def test_corner_case_4():
    params = {'adaptive_parsimony_scaling': 647.2371274931924, 'binary_operators': ['+', '-', '*'], 'complexity_of_variables': 2, 'constraints': {'*': (2, 1), 'square': 2, 'sqrt': 2, 'inv': 2}, 'crossover_probability': 0.02492145633410377, 'fraction_replaced': 0.002235225063443844, 'fraction_replaced_hof': 0.07533780585868693, 'interpretable_mode': True, 'logger_spec': None, 'maxdepth': 3, 'maxsize': 36, 'ncycles_per_iteration': 536, 'niterations': 11, 'optimize_probability': 0.7078781867923802, 'optimizer_f_calls_limit': 1229, 'output_directory': '/home/e23lerou/Documents/EquationDiscovery4ClimateImpact/data/run/edc8e0288d567d5315e2f17293418d9e', 'perturbation_factor': 0.10275045371494952, 'population_size': 42, 'populations': 4, 'probability_negate_constant': 0.029420840665321163, 'run_id': '669357efafe5ffd678815f6ebdcb5e23', 'timeout_in_seconds': 3600, 'topn': 10, 'tournament_selection_p': 0.7038487042682571, 'unary_operators': ['square', 'sqrt', 'inv(x) = 1/x'], 'warmup_maxsize_by': 0.33235592877205233, 'weight_add_node': 5.519664222209972, 'weight_delete_node': 0.34117290611362194, 'weight_do_nothing': 0.746485036643538, 'weight_insert_node': 0.1069031931866293, 'weight_mutate_constant': 0.25056611274194796, 'weight_mutate_operator': 1.558219468597955, 'weight_optimize': 0.10189345885829508, 'weight_randomize': 0.0036163899454910767, 'weight_rotate_tree': 19.435007512126443, 'weight_simplify': 0.0010482008385370033, 'weight_swap_operands': 1.0961264178207477}
    run_corner_case(0.25, ValidationSplit.QUANTILE_WITH_BINNING, params)

@pytest.mark.xfail
def test_corner_case_5():
    params = {'adaptive_parsimony_scaling': 1749.494188956888, 'binary_operators': ['+', '-', '*'], 'complexity_of_variables': 2, 'constraints': {'*': (2, 1), 'square': 2, 'sqrt': 2, 'inv': 2}, 'crossover_probability': 0.2444040755561205, 'fraction_replaced': 0.0008938288249710478, 'fraction_replaced_hof': 0.07999582659827606, 'interpretable_mode': True, 'logger_spec': None, 'maxdepth': 3, 'maxsize': 36, 'ncycles_per_iteration': 743, 'niterations': 37, 'optimize_probability': 0.8343522450717555, 'optimizer_f_calls_limit': 52812, 'output_directory': '/home/e23lerou/Documents/EquationDiscovery4ClimateImpact/data/run/9f666e4999cd5dd2309c3684dff946db', 'perturbation_factor': 0.6362362177340899, 'population_size': 41, 'populations': 7, 'probability_negate_constant': 0.05628161915279, 'run_id': '6025267500d7330786cc54d91d9a3f4f', 'timeout_in_seconds': 3600, 'topn': 5, 'tournament_selection_n': 13, 'tournament_selection_p': 0.7556416094454519, 'unary_operators': ['square', 'sqrt', 'inv(x) = 1/x'], 'warmup_maxsize_by': 0.7988027018497522, 'weight_add_node': 5.070147904779255, 'weight_delete_node': 0.1969594358122144, 'weight_do_nothing': 0.33590501268579814, 'weight_insert_node': 0.11030018071847747, 'weight_mutate_constant': 0.3245529082979231, 'weight_mutate_operator': 0.03574498276070293, 'weight_optimize': 0.24833333415649342, 'weight_randomize': 0.0007048656583144816, 'weight_rotate_tree': 12.060614027361142, 'weight_simplify': 0.017129796260070048, 'weight_swap_operands': 0.052928297286970946}
    run_corner_case(0.25, ValidationSplit.RANDOM, params)

@pytest.mark.xfail
def test_corner_case_6():
    params = {'adaptive_parsimony_scaling': 5841.541767434696, 'binary_operators': ['+', '-', '*'], 'complexity_of_variables': 2, 'constraints': {'*': (2, 1), 'square': 2, 'sqrt': 2, 'inv': 2}, 'crossover_probability': 0.2432870162201439, 'fraction_replaced': 0.0029053527949135117, 'fraction_replaced_hof': 0.5231736787347895, 'interpretable_mode': True, 'logger_spec': None, 'maxdepth': 2, 'maxsize': 24, 'ncycles_per_iteration': 176, 'niterations': 17, 'optimize_probability': 0.19589612249450167, 'optimizer_f_calls_limit': 23807, 'output_directory': '/home/e23lerou/Documents/EquationDiscovery4ClimateImpact/data/run/97abdb486b720bf96818d56094233668', 'perturbation_factor': 0.042675492580085767, 'population_size': 23, 'populations': 307, 'probability_negate_constant': 0.014566269683322771, 'run_id': 'e85343aa555efd9a6f75172b08d8ff6e', 'timeout_in_seconds': 3600, 'topn': 2, 'tournament_selection_n': 2, 'tournament_selection_p': 0.6218248106501165, 'unary_operators': ['square', 'sqrt', 'inv(x) = 1/x'], 'warmup_maxsize_by': 0.6650774434018913, 'weight_add_node': 1.327861581381353, 'weight_delete_node': 1.024108388899371, 'weight_do_nothing': 1.030295276903795, 'weight_insert_node': 0.09183139018351547, 'weight_mutate_constant': 0.33158320241563816, 'weight_mutate_operator': 1.3043783812418197, 'weight_optimize': 0.608357293045684, 'weight_randomize': 0.000943059188815957, 'weight_rotate_tree': 0.6744297424801324, 'weight_simplify': 0.0013369055141455876, 'weight_swap_operands': 0.4263014931858433}
    run_corner_case(0.2, ValidationSplit.MIDDLE, params)

@pytest.mark.xfail
def test_corner_case_7():
    params = {'adaptive_parsimony_scaling': 647.2371274931924, 'binary_operators': ['+', '-', '*'], 'complexity_of_variables': 2, 'constraints': {'*': (2, 1), 'square': 2, 'sqrt': 2, 'inv': 2}, 'crossover_probability': 0.02492145633410377, 'fraction_replaced': 0.002235225063443844, 'fraction_replaced_hof': 0.07533780585868693, 'interpretable_mode': True, 'logger_spec': None, 'maxdepth': 3, 'maxsize': 36, 'ncycles_per_iteration': 536, 'niterations': 11, 'optimize_probability': 0.7078781867923802, 'optimizer_f_calls_limit': 1229, 'output_directory': '/home/e23lerou/Documents/EquationDiscovery4ClimateImpact/data/run/c1665fb97c200b0491ba6c004339a442', 'perturbation_factor': 0.10275045371494952, 'population_size': 42, 'populations': 4, 'probability_negate_constant': 0.029420840665321163, 'run_id': '669357efafe5ffd678815f6ebdcb5e23', 'timeout_in_seconds': 3600, 'topn': 10, 'tournament_selection_p': 0.7038487042682571, 'unary_operators': ['square', 'sqrt', 'inv(x) = 1/x'], 'warmup_maxsize_by': 0.33235592877205233, 'weight_add_node': 5.519664222209972, 'weight_delete_node': 0.34117290611362194, 'weight_do_nothing': 0.746485036643538, 'weight_insert_node': 0.1069031931866293, 'weight_mutate_constant': 0.25056611274194796, 'weight_mutate_operator': 1.558219468597955, 'weight_optimize': 0.10189345885829508, 'weight_randomize': 0.0036163899454910767, 'weight_rotate_tree': 19.435007512126443, 'weight_simplify': 0.0010482008385370033, 'weight_swap_operands': 1.0961264178207477}
    run_corner_case(0.2, ValidationSplit.RANDOM, params)






