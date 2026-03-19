import numpy as np
import time

def bench(k):
    S = np.random.rand(k) + 0.1 # Ensure non-zero

    # Baseline
    start1 = time.perf_counter()
    for _ in range(100):
        S1 = np.diag(S)
        S1inv = np.linalg.inv(S1)
    end1 = time.perf_counter()

    # Optimized
    start2 = time.perf_counter()
    for _ in range(100):
        S1 = np.diag(S)
        S1inv_opt = np.diag(1.0 / S)
    end2 = time.perf_counter()

    # Correctness check
    assert np.allclose(S1inv, S1inv_opt)

    print(f"k={k}: Baseline = {(end1-start1):.5f}s, Optimized = {(end2-start2):.5f}s, Speedup = {(end1-start1)/(end2-start2):.2f}x")

bench(10)
bench(50)
bench(100)
bench(500)
bench(1000)
