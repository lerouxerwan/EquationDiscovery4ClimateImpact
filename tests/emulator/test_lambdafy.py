from emulator.utils_gaussian_fit import get_lambda_function_kwargs, get_lambda_function_list


def test_create_lambda_function_kwargs():
    s = "5 * x - 2 / y"
    f = get_lambda_function_kwargs(s)
    assert f(x=1, y=1) == 3
    assert f(x=2, y=2) == 9
    assert f(x=4, y=4) == 19.5

def test_create_lambda_function_list():
    s = "5 * x - 2 / y"
    f = get_lambda_function_list(s, ['x', 'y'])
    assert f([1, 1]) == 3
    assert f([2, 2]) == 9
    assert f([4, 4]) == 19.5


# def test_create_sympy_expr():
#     s = "5 * x - 2 / y"
#     expr = get_expr(s)
#     expr += x
#     print(expr)
# def get_expr(s: str) -> Expr:
#     x, y = symbols('x y')
#     return sympify(s)

