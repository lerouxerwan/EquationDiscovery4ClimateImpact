from sympy import symbols, exp, log, sqrt

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





