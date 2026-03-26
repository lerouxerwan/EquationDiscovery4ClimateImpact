from projects.paper.section_appendix.main_seasonal_repartition import main_plot_seasonal_repartition_three_pies
from projects.paper.section_data.main_plot_data import main_plot_data
from projects.paper.section_appendix.main_pareto_front_example import main_example_1d
from projects.paper.section_results.subsect_1_compare_optimization.main_compare_optimization_methods import \
    main_compare_optimization_methods
from projects.paper.section_results.subsect_1_compare_optimization.main_compare_optimization_methods_v2 import \
    main_compare_optimization_methods_v2
from projects.paper.section_results.subsect_1_compare_optimization.main_equation_table import main_equation_table
from projects.paper.section_results.subsect_2_compare_equations.main_four_features import main_four_features
from projects.paper.section_results.subsect_2_compare_equations.main_frequency_variable_name import \
    main_plot_frequency_variable
from projects.paper.section_results.subsect_2_compare_equations.main_plot_signed_frequency import \
    main_plot_signed_frequency
from projects.paper.section_results.subsect_3_analyze_best_equation.main_diagnosis_baseline_equation_with_four_features import \
    main_diagnosis_baseline_equation_with_four_selection_features
from projects.paper.section_results.subsect_3_analyze_best_equation.main_diagnosis_best_equation import \
    main_diagnosis_best_equation


def main_paper(show: bool):
    plot_functions = [
        # Data
        main_plot_data,
        # Appendix
        main_example_1d,
        # Results
        main_compare_optimization_methods_v2,
        # main_plot_frequency_variable,
        main_plot_signed_frequency,
        main_four_features,
        main_plot_seasonal_repartition_three_pies,
        # Diagnosis of equation for the last section
        # main_diagnosis_best_equation,
        main_diagnosis_baseline_equation_with_four_selection_features,
        # Last plot of results must be 'main_equation_table' to retrieve the output in the stdout/terminal
        main_equation_table,
    ]
    for plot_function in plot_functions:
        plot_function(show)

if __name__ == '__main__':
    main_paper(show=False)

