import numpy as np

def the_best_opt(S, V, unc):
    kmin, kmax = 0.00001, 10
    S2k = ((kmax - kmin) / 4) ** 2
    S1inv = np.linalg.inv(np.diag(S))
    S1inv_2 = S1inv**2
    S1inv_2_diag = np.diag(S1inv_2)
    Yemp = np.zeros(np.shape(V)[0])
    kt = []

    for s in range(0, np.shape(V)[1]):
        Y = Yemp.copy()
        Y[s] = 1
        Perrc = []
        for w in range(0, len(S)):
            S2E = (unc**2)[w]

            # Replaced loop with vectorized array dot products:
            P1i = np.sum((Y.T @ V[:, w + 1:]) ** 2) * S2k

            # The original:
            # SiyTvi = []
            # for i3 in range(1, w):
            #     Vi = V[:, i3]
            #     S2inv2YTVi = S1inv_2[i3 - 1, i3 - 1] * (Y.T @ Vi) ** 2
            #     SiyTvi.append(S2inv2YTVi)
            # P2i = np.sum(SiyTvi) * S2E

            if w > 1:
                P2i = np.sum(S1inv_2_diag[:w-1] * (Y.T @ V[:, 1:w]) ** 2) * S2E
            else:
                P2i = 0.0

            Perrc.append(P1i + P2i)
            k = np.argmin(Perrc)
            kt.append(k)
    return int(np.mean(kt))

def original_get_cutoff(S, V, unc):
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
            S2E = (unc**2)[w]
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

np.random.seed(42)
S = np.sort(np.random.rand(100))[::-1]
V = np.random.randn(100, 100)
unc = np.random.rand(100) * 0.1

print("Original:", original_get_cutoff(S, V, unc))
print("Safe Optimized:", the_best_opt(S, V, unc))

import timeit
t_orig = timeit.timeit(lambda: original_get_cutoff(S, V, unc), number=10)
t_opt = timeit.timeit(lambda: the_best_opt(S, V, unc), number=10)

print(f"Original time: {t_orig:.4f} seconds")
print(f"Safe Optimized time: {t_opt:.4f} seconds")
print(f"Speedup: {t_orig/t_opt:.2f}x")
