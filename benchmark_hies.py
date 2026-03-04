import time
import numpy as np

class MockSounding:
    def __init__(self, n):
        self.uncertainties = np.random.rand(n)

# We will mock the get_cutoff function
def get_cutoff_original(isounding, S, V, kmin=0.0001, kmax=10):
    kmin, kmax = 0.00001, 10
    S2k = ((kmax - kmin) / 4) ** 2
    S1inv = np.linalg.inv(np.diag(S))
    S1inv_2 = S1inv**2
    Yemp = np.zeros(np.shape(V)[0])
    kt = []
    for s in range(0, np.shape(V)[1]):
        Y = Yemp.copy()
        Y[s] = 1
        Perrc = []
        for w in range(0, len(S)):
            S2E = (isounding.uncertainties**2)[w]
            YtV_2 = []
            for i2 in range(w + 1, np.shape(V)[1]):
                Vi = V[:, i2]
                YtV_2.append((Y.T @ Vi) ** 2)
            P1i = np.sum(YtV_2) * S2k
            SiyTvi = []
            for i3 in range(1, w):
                Vi = V[:, i3]
                S2inv2YTVi = S1inv_2[i3 - 1, i3 - 1] * (Y.T @ Vi) ** 2
                SiyTvi.append(S2inv2YTVi)
            P2i = np.sum(SiyTvi) * S2E
            Perrc.append(P1i + P2i)
            k = np.argmin(Perrc)
            kt.append(k)
    return int(np.mean(kt))

def get_cutoff_optimized(isounding, S, V, kmin=0.0001, kmax=10):
    kmin, kmax = 0.00001, 10
    S2k = ((kmax - kmin) / 4) ** 2
    S1inv = np.linalg.inv(np.diag(S))
    S1inv_2 = S1inv**2

    v_shape_0, v_shape_1 = np.shape(V)

    Yemp = np.zeros(v_shape_0)
    kt = []
    for s in range(0, v_shape_1):
        Y = Yemp.copy()
        Y[s] = 1
        Perrc = []
        for w in range(0, len(S)):
            S2E = (isounding.uncertainties**2)[w]
            YtV_2 = []
            for i2 in range(w + 1, v_shape_1):
                Vi = V[:, i2]
                YtV_2.append((Y.T @ Vi) ** 2)
            P1i = np.sum(YtV_2) * S2k
            SiyTvi = []
            for i3 in range(1, w):
                Vi = V[:, i3]
                S2inv2YTVi = S1inv_2[i3 - 1, i3 - 1] * (Y.T @ Vi) ** 2
                SiyTvi.append(S2inv2YTVi)
            P2i = np.sum(SiyTvi) * S2E
            Perrc.append(P1i + P2i)
            k = np.argmin(Perrc)
            kt.append(k)
    return int(np.mean(kt))

def run_benchmark():
    n_w = 100
    n_v = 100
    isounding = MockSounding(n_w)
    S = np.random.rand(n_w) + 1.0  # Avoid division by zero
    V = np.random.rand(n_w, n_v)

    start = time.time()
    res1 = get_cutoff_original(isounding, S, V)
    t1 = time.time() - start

    start = time.time()
    res2 = get_cutoff_optimized(isounding, S, V)
    t2 = time.time() - start

    print(f"Original: {t1:.4f}s")
    print(f"Optimized: {t2:.4f}s")
    print(f"Results match: {res1 == res2}")

run_benchmark()
