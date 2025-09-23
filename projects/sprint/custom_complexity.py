

# Missing docstring for Symbolics.get_variables!. Check Documenter's build log for details.
# la fonction "arguments" ne semble pas fonctionner sur mes nodes. En gros ca ne retourne qu'un seul term dont la valeur est l'expression d'origine
# et donc les contraintes (celle du nombre de variable et celle de la saisonalité) était appliqué sur l expression entière
# Bref il faut comprendre comment faire marcher "arguments" dans mon cas. Est-ce que je dois faire un "convert" avant ?
complexity_mapping = """
using Symbolics
using SymbolicUtils
function variable_sparsity_complexity(expression)
    # Classical complexity
    complexity = 0
    for node in get_tree(expression)
        complexity += 1
    end
    # Penalize any term with more than 3 features
    println("start loop on terms")
    # println(complexity)
    # terms = arguments(Symbolics.unwrap(expression), +)
    terms = Symbolics.arguments(Symbolics.unwrap(expression), +)
    # terms = arguments(expression, -)
    println(length(terms))
    dump(terms)
    for term in terms
        features = Set{UInt16}()
        local_complexity = 0
        for node in get_tree(term)
            local_complexity += 1
            if node.degree == 0 && !node.constant
                push!(features, node.feature)
            end
        end
        if local_complexity < complexity
            println("here")
            # println(local_complexity)
        end
        seasons = Set{UInt8}()
        for feature in features
            push!(seasons, feature % 4)
        end
        if (length(features) >= 3) || (length(seasons) >= 2)
            # println("Features")
            # for feature in features        
            #     println(feature)
            # end
            # println("Seasons")
            # for season in seasons
            #     println(season)        
            # end
            complexity = 31
            break # End quickly the function if there is a term with 3 features
        end
    end
    return complexity
end
"""


complexity_mapping_basic = """
function custom_complexity(expr::AbstractExpression)
    return 5
end
"""

complexity_mapping_level1 = """
function variable_sparsity_complexity(expression)
    tree = get_tree(expression)  # (for template expressions, would need to do something more complex)
    complexity = 0
    for node in tree
        complexity += 1
    end
    return complexity
end
"""

complexity_mapping_level2 = """
function variable_sparsity_complexity(expression)
    tree = get_tree(expression)  # (for template expressions, would need to do something more complex)
    complexity = 0
    for node in tree
        T = typeof(node)
        for (name, typ) in zip(fieldnames(T), T.types)
            println("type of the fieldname $name is $typ")
        end
        complexity += 1
        if node.degree == 0 && !node.constant
            println(node.feature)
        end
        if node.constant
            println(node.degree)
            println(node.val)
        end
        println(node.degree)
        println(node.op)
    end
    return complexity
end
"""

# complexity_mapping = """
# function custom_complexity(expr::AbstractExpression)
#
#     # Classical definition of complexity
#     nb_constantes = 0
#     nb_variables = 0
#     nb_operateurs = 0
#
#     for node in Iterators.PostOrderDFS(expr)
#         if node isa AbstractExpressionNode
#             if node.op == :constant
#                 nb_constantes += 1
#             elseif node.op == :variable
#                 nb_variables += 1
#             else
#                 # Tout autre nœud est considéré comme un opérateur
#                 nb_operateurs += 1
#             end
#         end
#     end
#
#     complexity = nb_constantes + nb_variables + nb_operators
#
#     return complexity
# end
# """


"""


    # Add penalization when there is a term with three variables

    function count_variables(node::AbstractExpressionNode, variables::Set{Symbol})
        if node isa AbstractExpressionNode
            if node.op == :variable
                push!(variables, node.val)
            else
                for child in node.children
                    count_variables(child, variables)
                end
            end
        end
        return variables
    end

    for node in expr
        if node isa AbstractExpressionNode
            variables = Set{Symbol}()
            count_variables(node, variables)
            if length(variables) >= 3
                complexity += 20
            end
        end
    end
"""