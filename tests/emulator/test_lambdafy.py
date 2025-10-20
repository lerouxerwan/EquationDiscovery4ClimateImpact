from typing import Callable


def get_lambda_function(s: str) -> Callable:
    return lambda **kwargs: eval(s, {}, kwargs)



def test_create_lambda_function():
    s = "5 * x - 2 / y"
    f = get_lambda_function(s)
    assert f(x=1, y=1) == 3
    assert f(x=2, y=2) == 9
    assert f(x=4, y=4) == 19.5

# def test_create_sympy_expr():
#     s = "5 * x - 2 / y"
#     expr = get_expr(s)
#     expr += x
#     print(expr)
# def get_expr(s: str) -> Expr:
#     x, y = symbols('x y')
#     return sympify(s)

