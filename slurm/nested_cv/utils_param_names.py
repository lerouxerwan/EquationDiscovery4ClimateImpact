
param_names = ['maxsize', 'warmup_maxsize_by', 'unary_operators', 'populations', 'population_size', 'ncycles_per_iteration',
     'topn', 'optimizer_f_calls_limit', 'optimize_probability', 'tournament_selection_p', 'tournament_selection_n',
     'weight_optimize', 'adaptive_parsimony_scaling', 'fraction_replaced', 'fraction_replaced_hof', 'weight_add_node',
     'weight_insert_node', 'weight_delete_node', 'weight_do_nothing', 'weight_mutate_constant',
     'weight_mutate_operator', 'weight_swap_operands', 'weight_rotate_tree', 'weight_randomize', 'weight_simplify',
     'crossover_probability', 'perturbation_factor', 'probability_negate_constant']

if __name__ == '__main__':
    print(len(param_names))