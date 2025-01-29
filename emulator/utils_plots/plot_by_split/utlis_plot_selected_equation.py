from matplotlib.axes import Axes
from sympy import Expr

from emulator.utils_plots.plot_by_split.utils_equation_str import get_equation_str


def get_true_label_and_predicted_label(label: str, remove_units=False) -> tuple[str, str]:
    return (get_label(f"Reference {label}", remove_units),
            get_label(f"Predicted {label}", remove_units))

def get_label(label: str, remove_units=False) -> str:
    unit = '' if remove_units else get_unit(label)
    return label.split('(')[0] + unit


def get_unit(label: str) -> str:
    return '(' + label.split('(')[-1].replace(' ', '')



def add_equation(ax: Axes, expr: Expr):
    coef = 0.95
    ax.annotate(f'Equation found: {get_equation_str(expr)}', xy=(0.02, 0.95),
                xycoords='axes fraction', textcoords='offset points', size=7,
                bbox=dict(boxstyle="round", fc=(coef, coef, coef), ec="none"))

