
function test_iterations(n)
    x = 1.0  # Float64
    for i in 1:n
        x = x + 0.1  # Opération sensible aux erreurs d'arrondi
        println("Itération $i: x = $x")
    end
    return x
end

# Exécutez sur chaque machine et comparez les sorties
test_iterations(1000)