from sympy import symbols, sqrt

from emulator.utils_variable_names import get_variable_signed_names

if __name__ == '__main__':
    x, y = symbols('x y')
    expr = - 7 / (x ** 2) + 8 / sqrt(y)
    variable_signed_names = get_variable_signed_names(expr)
    assert set(variable_signed_names) == {'+x', '-y'}, variable_signed_names