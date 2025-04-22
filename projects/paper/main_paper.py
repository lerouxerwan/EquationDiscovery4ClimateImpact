from projects.paper.section_data.main_plot_data import main_plot_data
from projects.paper.section_methodology.main_pareto_front_example import main_example_1d
from projects.paper.section_results.main_plot_results import main_plot_results


def main_paper(show: bool):
    plot_functions = [
        # Data
        main_plot_data,
        # Methodology
        main_example_1d,
        # Results
        main_plot_results,
    ]
    for plot_function in plot_functions:
        plot_function(show)

if __name__ == '__main__':
    main_paper(show=False)

