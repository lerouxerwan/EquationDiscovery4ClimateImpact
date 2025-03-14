import pandas as pd
from matplotlib import pyplot as plt

from data.utils_search.search_experiment import SearchExperiment
from utils.utils_plot import show_or_save_plot


def plot_diagnosis_search(search_experiment: SearchExperiment, show: bool = False):
    plot_diagnosis_search_1d(search_experiment, show)

def plot_diagnosis_search_1d(search_experiment: SearchExperiment, show: bool):
    """Plot the variation of RMSE validation for each hyperparameter in the param_grid"""
    df = search_experiment.df_cv_results_ranked_and_augmented
    params_list = df['params'].to_list()
    metric_name = 'RMSE_validation'
    for param_name in search_experiment.get_combinations_of_param_names_in_param_grid(nb_elements=1):
        param_name = param_name[0]
        ax = plt.gca()
        min_loss_list = []
        param_values = [params[param_name] for params in params_list]
        sorted_param_values = sorted(list(set(param_values)))
        for sorted_param_value in sorted_param_values:
            ind = pd.Series(index=df.index, data=[v == sorted_param_value for v in param_values])
            min_loss = df.loc[ind, metric_name].min()
            min_loss_list.append(min_loss)
        ax.plot(sorted_param_values, min_loss_list)
        create_label = lambda s: ' '.join([w.capitalize() for w in s.split('_')])
        ax.set_xlabel(create_label(param_name))
        ax.set_ylabel(create_label(metric_name))
        show_or_save_plot(f"diagnosis_1D_{param_name}", show)

if __name__ == '__main__':
    search_path = ("/home/e23lerou/Documents/EquationDiscovery4ClimateImpact/data/search/6624636351014803864/"
                   "RandomizedSearchCV_10_fra0.06140000000000002_nit10_ada520.0_2080.0_fra0.0_0.1_nit5_20_sca2_thr1.0005")
    plot_diagnosis_search_1d(SearchExperiment(search_path), show=True)