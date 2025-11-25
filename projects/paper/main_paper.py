from projects.paper.section_data.main_plot_data import main_plot_data
from projects.paper.section_appendix.main_pareto_front_example import main_example_1d


def main_paper(show: bool):
    plot_functions = [
        # Data
        main_plot_data,
        # Methodology
        main_example_1d,
        # Results
    ]
    for plot_function in plot_functions:
        plot_function(show)

if __name__ == '__main__':
    main_paper(show=False)

