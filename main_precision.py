import numpy as np

def test_iterations(n):
    x = np.float64(1.0)
    for i in range(n):
        x += 0.1
        print(f"Itération {i}: x = {x}")
    return x

test_iterations(1000)