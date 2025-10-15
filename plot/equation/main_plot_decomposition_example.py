from sympy import symbols

from data.utils_dataset.npp_season_v1 import get_dataset
from plot.equation.plot_decomposition import _plot_decomposition

if __name__ == '__main__':
    dataset = get_dataset()
    SST_DJF, SeaSurfaceStericHeight_DJF, SSS_MAM, Shortwave_DJF, SST_MAM = symbols('SST_DJF, SeaSurfaceStericHeight_DJF, SSS_MAM, Shortwave_DJF, SST_MAM')
    expr = -1.0864899*SST_DJF + 42.891544*SeaSurfaceStericHeight_DJF**2 + 2550.9436*(SSS_MAM + 0.010449366*Shortwave_DJF)/SST_MAM

    print(dataset.X_variable_names)
    X = dataset.X_train
    years = dataset.years_train
    _plot_decomposition(expr, X, dataset.X_variable_names, years)