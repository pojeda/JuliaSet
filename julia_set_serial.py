""" 

Pedro Ojeda Feb. 2022 

"""
import numpy as np
import matplotlib.pyplot as plt
import time

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
    N = 5000
    max_iter = 300
    c = -0.7 + 0.27015j
    x = np.linspace(-1.5, 1.5, N)
    y = np.linspace(-1.5, 1.5, N)

    start = time.perf_counter()
    image = julia(c, x, y, max_iter)
    elapsed = time.perf_counter() - start

    print(f"Serial time: {elapsed:.2f} seconds")
    plt.imshow(np.log1p(image), extent=(-1.5, 1.5, -1.5, 1.5),
               origin="lower", cmap="jet")
    plt.colorbar()
    plt.title(f"Serial: {elapsed:.2f} s")
    plt.savefig("julia_serial.png", dpi=150)
    plt.close()
