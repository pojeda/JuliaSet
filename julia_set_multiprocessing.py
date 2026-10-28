import time
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt


def julia(c, x, y, max_iter):
    X, Y = np.meshgrid(x, y)
    z = X + 1j * Y
    counts = np.zeros(z.shape, dtype=np.int32)
    active = np.ones(z.shape, dtype=bool)

    for _ in range(max_iter):
        z[active] = z[active] ** 2 + c
        counts[active] += 1
        active &= np.abs(z) <= 2
        if not active.any():
            break

    return counts


if __name__ == "__main__":
    N = 3000
    max_iter = 300
    workers = 4
    c = -0.7 + 0.27015j
    x = np.linspace(-1.5, 1.5, N)
    y = np.linspace(-1.5, 1.5, N)

    # Split the rows among the worker processes.
    blocks = np.array_split(y, workers)
    tasks = [(c, x, block, max_iter) for block in blocks]

    start = time.perf_counter()
    with Pool(workers) as pool:
        results = pool.starmap(julia, tasks)
    image = np.vstack(results)
    elapsed = time.perf_counter() - start

    print(f"Multiprocessing time ({workers} workers): {elapsed:.2f} seconds")
    plt.imshow(np.log1p(image), extent=(-1.5, 1.5, -1.5, 1.5),
               origin="lower", cmap="jet")
    plt.colorbar()
    plt.title(f"Multiprocessing ({workers} workers): {elapsed:.2f} s")
    plt.savefig("julia_multiprocessing.png", dpi=150)
    plt.close()
