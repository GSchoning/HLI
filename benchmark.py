import timeit
import numpy as np

def original_method(p_input, n_layers):
    z_vec = np.zeros(n_layers)
    for i in range(n_layers):
        z_vec[i] = p_input.get(f"z_{i:02d}", 0.0)
    return z_vec

def proposed_method(p_input, n_layers):
    z_vec = np.array([p_input.get(f"z_{i:02d}", 0.0) for i in range(n_layers)])
    return z_vec

if __name__ == "__main__":
    n_layers = 50
    p_input = {f"z_{i:02d}": np.random.randn() for i in range(n_layers)}

    # Verify correctness
    assert np.allclose(original_method(p_input, n_layers), proposed_method(p_input, n_layers))

    # Benchmark
    N = 100000
    orig_time = timeit.timeit(lambda: original_method(p_input, n_layers), number=N)
    prop_time = timeit.timeit(lambda: proposed_method(p_input, n_layers), number=N)

    print(f"Original method: {orig_time:.6f} seconds")
    print(f"Proposed method: {prop_time:.6f} seconds")
    print(f"Speedup: {orig_time / prop_time:.2f}x")
