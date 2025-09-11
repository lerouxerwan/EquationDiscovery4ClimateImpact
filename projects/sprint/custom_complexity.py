
complexity_mapping_basic = """
function custom_complexity(expr::AbstractExpression)
    return 5
end
"""

complexity_mapping = """
function custom_complexity(expr::AbstractExpression)
   
    # Classical definition of complexity
    nb_constantes = 0
    nb_variables = 0
    nb_operateurs = 0

    for node in Iterators.PostOrderDFS(expr)
        if node isa AbstractExpressionNode
            if node.op == :constant
                nb_constantes += 1
            elseif node.op == :variable
                nb_variables += 1
            else
                # Tout autre nœud est considéré comme un opérateur
                nb_operateurs += 1
            end
        end
    end
    
    complexity = nb_constantes + nb_variables + nb_operators
    
    return complexity
end
"""


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