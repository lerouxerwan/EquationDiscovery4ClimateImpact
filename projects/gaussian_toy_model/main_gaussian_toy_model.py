from projects.gaussian_toy_model.gaussian_toy_model import GaussianToyModel
from projects.gaussian_toy_model.plot_sensitivity_analysis import plot_sensitivity_analysis


def main_gaussian_toy_model_one_plot(fast: bool, show: bool):
    niterations = 1 if fast else 100
    model = GaussianToyModel(mu_degree=2, sigma_degree=0, nb_samples=1000, params={"niterations": niterations})
    model.plot(show=show)

def main_gaussian_toy_model_all_plots(fast: bool, show: bool):
    niterations = 1 if fast else 100
    for mu_degree in [1, 2]:
        for sigma_degree in [0, 1]:
            model = GaussianToyModel(mu_degree=mu_degree, sigma_degree=sigma_degree,
                                     nb_samples=1000, params={"niterations": niterations})
            model.plot(show=show)


def main_gaussian_toy_model_1_three_detailed_plot(fast: bool, show: bool):
    niterations = 1 if fast else 100
    for subplot in [0, 1, 2]:
            model = GaussianToyModel(mu_degree=1, sigma_degree=0,
                                     nb_samples=1000, params={"niterations": niterations})
            model.plot(show=show, subplot=subplot)

def main_sensitivity_analysis_nb_datapoints(fast: bool, show: bool):
    for mu_degree in [1, 2]:
        for sigma_degree in [0, 1]:
            niterations = 100
            nb_samples_list = [100, 300, 500, 700, 1000]
            if fast:
                nb_samples_list = nb_samples_list[-1:]
            plot_sensitivity_analysis(mu_degree, sigma_degree, nb_samples_list, niterations, show)



if __name__ == '__main__':
    # main_gaussian_toy_model_all_plots(fast=False, show=False)
    main_gaussian_toy_model_1_three_detailed_plot(fast=False, show=False)
    # main_sensitivity_analysis_nb_datapoints(fast=False)
