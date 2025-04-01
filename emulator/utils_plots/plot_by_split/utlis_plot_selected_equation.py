from matplotlib.axes import Axes
from sympy import Expr

from emulator.utils_plots.plot_by_split.utils_equation_str import get_equation_str


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



def add_equation(ax: Axes, expr: Expr):
    coef = 0.95
    ax.annotate(f'Equation with rounded coefficients: {get_equation_str(expr)}', xy=(0.02, 0.9),
                xycoords='axes fraction', textcoords='offset points', size=7,
                bbox=dict(boxstyle="round", fc=(coef, coef, coef), ec="none"))

