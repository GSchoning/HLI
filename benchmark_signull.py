import timeit
import numpy as np

def run_ies_forward_original(p_input, n_layers):
    z_vec = np.zeros(n_layers)
    for i in range(n_layers):
        z_vec[i] = p_input.get(f"z_{i:02d}", 0.0)
    return z_vec

def run_ies_forward_new(p_input, n_layers):
    z_vec = np.array([p_input.get(f"z_{i:02d}", 0.0) for i in range(n_layers)])
    return z_vec

if __name__ == '__main__':
    n_layers = 100
    p_input = {f"z_{i:02d}": float(i) for i in range(n_layers)}

    assert np.allclose(run_ies_forward_original(p_input, n_layers), run_ies_forward_new(p_input, n_layers))

    print("Benchmarking n_layers=100")
    setup = "from __main__ import run_ies_forward_original, run_ies_forward_new, p_input, n_layers"

    orig_time = timeit.timeit("run_ies_forward_original(p_input, n_layers)", setup=setup, number=10000)
    new_time = timeit.timeit("run_ies_forward_new(p_input, n_layers)", setup=setup, number=10000)

    print(f"Original time: {orig_time:.6f} s")
    print(f"New time:      {new_time:.6f} s")
    print(f"Improvement:   {orig_time / new_time:.2f}x")

    n_layers = 1000
    p_input = {f"z_{i:02d}": float(i) for i in range(n_layers)}

    assert np.allclose(run_ies_forward_original(p_input, n_layers), run_ies_forward_new(p_input, n_layers))

    print("\nBenchmarking n_layers=1000")
    setup = "from __main__ import run_ies_forward_original, run_ies_forward_new, p_input, n_layers"

    orig_time = timeit.timeit("run_ies_forward_original(p_input, n_layers)", setup=setup, number=1000)
    new_time = timeit.timeit("run_ies_forward_new(p_input, n_layers)", setup=setup, number=1000)

    print(f"Original time: {orig_time:.6f} s")
    print(f"New time:      {new_time:.6f} s")
    print(f"Improvement:   {orig_time / new_time:.2f}x")
