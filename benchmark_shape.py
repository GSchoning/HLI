import numpy as np
import time

def get_cutoff_original(S, V, uncertainties):
    kmin, kmax = 0.00001, 10
    S2k = ((kmax - kmin) / 4) ** 2
    S1inv = np.linalg.inv(np.diag(S))
    S1inv_2 = S1inv**2
    Yemp = np.zeros(np.shape(V)[0])
    kt = []

    # Simulate the loop
    for s in range(0, np.shape(V)[1]):
        Y = Yemp.copy(); Y[s] = 1
        Perrc = []
        for w in range(0, len(S)):
            S2E = (uncertainties**2)[w]
            YtV_2 = []
            for i2 in range(w + 1, np.shape(V)[1]):
                Vi = V[:, i2]; YtV_2.append((Y.T @ Vi) ** 2)
            P1i = np.sum(YtV_2) * S2k
            SiyTvi = []
            for i3 in range(1, w):
                Vi = V[:, i3]; S2inv2YTVi = S1inv_2[i3 - 1, i3 - 1] * (Y.T @ Vi) ** 2
                SiyTvi.append(S2inv2YTVi)
            P2i = np.sum(SiyTvi) * S2E
            Perrc.append(P1i + P2i)
            k = np.argmin(Perrc)
            kt.append(k)
    return int(np.mean(kt))

def get_cutoff_optimized(S, V, uncertainties):
    kmin, kmax = 0.00001, 10
    S2k = ((kmax - kmin) / 4) ** 2
    S1inv = np.linalg.inv(np.diag(S))
    S1inv_2 = S1inv**2

    V_shape_0 = V.shape[0]
    V_shape_1 = V.shape[1]

    Yemp = np.zeros(V_shape_0)
    kt = []

    for s in range(0, V_shape_1):
        Y = Yemp.copy(); Y[s] = 1
        Perrc = []
        for w in range(0, len(S)):
            S2E = (uncertainties**2)[w]
            YtV_2 = []
            for i2 in range(w + 1, V_shape_1):
                Vi = V[:, i2]; YtV_2.append((Y.T @ Vi) ** 2)
            P1i = np.sum(YtV_2) * S2k
            SiyTvi = []
            for i3 in range(1, w):
                Vi = V[:, i3]; S2inv2YTVi = S1inv_2[i3 - 1, i3 - 1] * (Y.T @ Vi) ** 2
                SiyTvi.append(S2inv2YTVi)
            P2i = np.sum(SiyTvi) * S2E
            Perrc.append(P1i + P2i)
            k = np.argmin(Perrc)
            kt.append(k)
    return int(np.mean(kt))

# Create dummy data
np.random.seed(42)
N = 100
M = 50
S = np.random.rand(M)
V = np.random.rand(N, M)
uncertainties = np.random.rand(M)

# Measure original
start = time.time()
for _ in range(50):
    get_cutoff_original(S, V, uncertainties)
end = time.time()
print(f"Original: {end - start:.5f} s")

# Measure optimized
start = time.time()
for _ in range(50):
    get_cutoff_optimized(S, V, uncertainties)
end = time.time()
print(f"Optimized: {end - start:.5f} s")
