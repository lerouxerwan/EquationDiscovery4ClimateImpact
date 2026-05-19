from projects.gaussian_toy_model.gaussian_toy_model import GaussianToyModel


def main_gaussian_toy_model_one_plot(fast: bool):
    niterations = 1 if fast else 100
    model = GaussianToyModel(mu_degree=2, sigma_degree=0, nb_samples=1000, params={"niterations": niterations})
    model.plot()

def main_gaussian_toy_model_all_plots(fast: bool):
    niterations = 1 if fast else 100
    for mu_degree in [1, 2]:
        for sigma_degree in [0, 1]:
            model = GaussianToyModel(mu_degree=mu_degree, sigma_degree=sigma_degree,
                                     nb_samples=1000, params={"niterations": niterations})
            model.plot(show=False)


def main_gaussian_toy_model_1_three_detailed_plot(fast: bool):
    niterations = 1 if fast else 100
    for subplot in [0, 1, 2]:
            model = GaussianToyModel(mu_degree=1, sigma_degree=0,
                                     nb_samples=1000, params={"niterations": niterations})
            model.plot(show=False, subplot=subplot)



if __name__ == '__main__':
    # main_gaussian_toy_model_all_plots(fast=False)
    main_gaussian_toy_model_1_three_detailed_plot(fast=False)
