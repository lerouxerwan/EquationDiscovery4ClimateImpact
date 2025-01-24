from matplotlib.axes import Axes
from sympy import Expr

from emulator.utils_plots.utils_equation_str import get_equation_str


def get_labels(target_label: str, remove_units=False) -> list[str]:
    labels = [f"True {target_label}", f"Predicted {target_label}"]
    if remove_units:
        labels = [label.split('(')[0] for label in labels]
    return labels

def add_equation(ax: Axes, expr: Expr):
    coef = 0.95
    ax.annotate(f'Equation found: {get_equation_str(expr)}', xy=(0.02, 0.95),
                xycoords='axes fraction', textcoords='offset points', size=7,
                bbox=dict(boxstyle="round", fc=(coef, coef, coef), ec="none"))

