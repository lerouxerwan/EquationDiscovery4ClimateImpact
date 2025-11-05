from emulator.utils_gaussian_fit import get_lambda_function_list



def test_create_lambda_function_list():
    s = "5 * x - 2 / y"
    f = get_lambda_function_list(s, ['x', 'y'])
    assert f([1, 1]) == 3
    assert f([2, 2]) == 9
    assert f([4, 4]) == 19.5



