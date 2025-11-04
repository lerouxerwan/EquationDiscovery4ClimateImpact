from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sympy import Expr



def get_true_label_and_predicted_label(label: str, remove_units=False) -> list[str]:
    return [get_label(f"{prefix} of\n{uncapitalize(label)}", remove_units) for prefix in get_true_and_predicted_prefix()]

def get_true_and_predicted_prefix():
    return "Ground truth values", "Predicted values"

def get_true_and_predicted_label():
    return "Ground truth", "Prediction"

def uncapitalize(s):
    return s[:1].lower() + s[1:]


def get_label(label: str, remove_units=False) -> str:
    unit = '' if remove_units else get_unit(label)
    return label.split('(')[0] + unit


def get_unit(label: str) -> str:
    return '(' + label.split('(')[-1].replace(' ', '')


def add_equation(equation: str):
    plt.suptitle(f'Equation: {equation}')

