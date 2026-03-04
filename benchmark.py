import timeit

setup = '''
cols = [f"GateTimeLM_{i}" for i in range(100)] + [f"GateTimeHM_{i}" for i in range(100)] + [f"OtherCol_{i}" for i in range(100)]
'''

test_filter_lambda = '''
LMCols = list(filter(lambda k: "GateTimeLM" in k, cols))
HMCols = list(filter(lambda k: "GateTimeHM" in k, cols))
'''

test_list_comp = '''
LMCols = [k for k in cols if "GateTimeLM" in k]
HMCols = [k for k in cols if "GateTimeHM" in k]
'''

n = 10000
t1 = timeit.timeit(test_filter_lambda, setup=setup, number=n)
t2 = timeit.timeit(test_list_comp, setup=setup, number=n)

print(f"filter + lambda: {t1:.4f} seconds")
print(f"list comp:       {t2:.4f} seconds")
print(f"Improvement:     {(t1 - t2) / t1 * 100:.2f}%")
