import time
import numpy as np
import sys
sys.path.append('.')
from libraries.SigNULL import get_cutoff

class DummySounding:
    def __init__(self, uncertainties):
        self.uncertainties = uncertainties

def run_benchmark():
    np.random.seed(42)
    # create some large inputs
    S = np.random.rand(100) + 0.1  # avoid zero
    # V's columns (second dimension) define loop iterations over s
    V = np.random.rand(200, 100)
    uncertainties = np.random.rand(100)
    isounding = DummySounding(uncertainties)

    start_time = time.time()
    for _ in range(20):
        res = get_cutoff(isounding, S, V)
    end_time = time.time()

    print(f"Benchmark took: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    run_benchmark()
