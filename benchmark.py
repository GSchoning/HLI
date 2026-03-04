import numpy as np
import time

def get_cutoff_original(S):
    # original logic, just simulating the S1inv and S1inv_2 parts
    S1inv = np.linalg.inv(np.diag(S))
    S1inv_2 = S1inv**2
    return S1inv_2

def get_cutoff_optimized(S):
    # optimized logic
    S1inv = np.diag(1.0 / S)
    S1inv_2 = S1inv**2
    return S1inv_2

sizes = [100, 500, 1000]

for n in sizes:
    S = np.random.rand(n)

    start = time.time()
    for _ in range(10):
        get_cutoff_original(S)
    t_orig = time.time() - start

    start = time.time()
    for _ in range(10):
        get_cutoff_optimized(S)
    t_opt = time.time() - start

    print(f"Size {n}: original = {t_orig:.4f}s, optimized = {t_opt:.4f}s, speedup = {t_orig/t_opt:.2f}x")

    assert np.allclose(get_cutoff_original(S), get_cutoff_optimized(S))
