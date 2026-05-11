from projects.gaussian_toy_model.gaussian_toy_model import GaussianToyModel


def main_gaussian_toy_model():
    model = GaussianToyModel(mu_degree=1, sigma_degree=0, nb_samples=1000)
    model.plot()

if __name__ == '__main__':
    main_gaussian_toy_model()