from matplotlib import pyplot as plt

from projects.gaussian_toy_model.gaussian_toy_model import GaussianToyModel
from utils.utils_plot import show_or_save_plot


def plot_sensitivity_analysis(mu_degree: int, sigma_degree: int, nb_samples_list: list[int], niterations: int):
    """
    Plot how the log likelihood changes with respect to the number of samples.
    To have a fair comparison, we compute the log likelihood on the same set of samples (nb_samples_list[-1])
    """
    # Start plot
    ax = plt.gca()
    values = []
    labels = []
    nb_samples_plot = []
    # Reference data for the plot
    model = GaussianToyModel(mu_degree=mu_degree, sigma_degree=sigma_degree,
                             nb_samples=nb_samples_list[-1], params={"niterations": niterations})
    X, y = model.X_and_y
    # Extract log likelihood and label for the groundtruth
    # This is probably quite complicated to plot
    # Extract log likelihood and label for each number of samples
    for nb_samples in nb_samples_list:
        model = GaussianToyModel(mu_degree=mu_degree, sigma_degree=sigma_degree,
                                 nb_samples=nb_samples, params={"niterations": niterations})
        log_likelihood = model.emulator.compute_loss(X, y)
        if log_likelihood < 100:
            nb_samples_plot.append(nb_samples)
            values.append(log_likelihood)
            labels.append(model.get_label_discovered())
    # Plot bar
    ax.set_title(model.get_label_ground_truth())
    ax.bar(nb_samples_plot, height=values, width=100)
    print(values)
    print(labels)
    ax.set_xlabel("Number of samples")
    ax.set_ylabel('Log likelihood for the discovered equation')
    nb_samples_list_str = "_".join([str(i) for i in nb_samples_list])
    plot_name = "sensitivity_analysis_{}_{}_{}".format(mu_degree, sigma_degree, nb_samples_list_str)
    show_or_save_plot(plot_name=plot_name, show=False)
