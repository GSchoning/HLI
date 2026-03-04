import timeit
import numpy as np

setup_code = """
import numpy as np
np.random.seed(0)
S = np.random.rand(100)
"""

test_code_old = """
S1inv = np.linalg.inv(np.diag(S))
S1inv_2 = S1inv**2
"""

test_code_new = """
S1inv = np.diag(1.0 / S)
S1inv_2 = S1inv**2
"""

time_old = timeit.timeit(test_code_old, setup=setup_code, number=10000)
time_new = timeit.timeit(test_code_new, setup=setup_code, number=10000)

print(f"Old approach time: {time_old:.5f} seconds")
print(f"New approach time: {time_new:.5f} seconds")
print(f"Speedup: {time_old/time_new:.2f}x")
