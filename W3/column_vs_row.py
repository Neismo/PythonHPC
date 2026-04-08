from time import perf_counter as time

import numpy as np

# Rows vs. Columns
# ns = [100, 1000, 10000]  # <-- From the book
ns = np.round(np.logspace(1, 4.5, 30))
trows = []
tcols = []
n_repeat = int(1e3)

for n in ns:
    n = int(n)
    mat = np.random.rand(n, n)

    trow = time()
    for _ in range(n_repeat):
        mat[0, :] * 1.01
    trow = time() - trow

    tcol = time()
    for _ in range(n_repeat):
        mat[:, 0] * 1.01
    tcol = time() - tcol

    trows.append(trow / n_repeat)
    tcols.append(tcol / n_repeat)

print(mat.dtype)
print('ns =', list(ns))
print('trows =', trows)
print('tcols =', tcols)