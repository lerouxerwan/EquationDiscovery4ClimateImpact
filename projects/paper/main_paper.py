from projects.paper.section_data.main_plot_data import main_plot_data
from projects.paper.section_appendix.main_pareto_front_example import main_example_1d
from projects.paper.section_results.subsect_1_compare_optimization.main_compare_optimization_methods import \
    main_compare_optimization_methods
from projects.paper.section_results.subsect_1_compare_optimization.main_equation_table import main_equation_table
from projects.paper.section_results.subsect_2_compare_equations.main_frequency_variable_name import \
    main_plot_frequency_variable
from projects.paper.section_results.subsect_3_analyze_best_equation.main_diagnosis_best_equation import \
    main_diagnosis_best_equation


def main_paper(show: bool):
    plot_functions = [
        # Data
        main_plot_data,
        # Methodology
        main_example_1d,
        # Results
        # Subsect 1
        main_compare_optimization_methods,
        main_equation_table,
        # Subsect 2
        main_plot_frequency_variable,
        # Subsect 3
        main_diagnosis_best_equation,
    ]
    for plot_function in plot_functions:
        plot_function(show)

if __name__ == '__main__':
    main_paper(show=False)

