from sympy import symbols, sqrt

from emulator.utils_variable_names import get_variable_signed_names


def test_variable_signed_names():
    x, y = symbols('x y')
    expr = 1 + x - x
    assert set(get_variable_signed_names(expr)) == set()
    expr = 1 + 2 * x - 3 * y
    assert set(get_variable_signed_names(expr)) == {'+x', '-y'}
    expr = 4 * (x ** 2) + 5 / y + 6
    assert set(get_variable_signed_names(expr)) == {'+x', '-y'}
    expr = - 7 / (x ** 2) + 8 / sqrt(y)
    assert set(get_variable_signed_names(expr)) == {'+x', '-y'}
