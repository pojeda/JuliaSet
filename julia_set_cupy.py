import time
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

N = 1000
max_iter = 300
c = -0.7 + 0.27015j

x = cp.linspace(-1.5, 1.5, N)
X, Y = cp.meshgrid(x, x)
z = X + 1j * Y
counts = cp.zeros((N, N), dtype=cp.int32)
active = cp.ones((N, N), dtype=cp.bool_)

start = time.perf_counter()
for _ in range(max_iter):
    z = cp.where(active, z * z + c, z)
    counts += active
    active &= cp.abs(z) <= 2

counts_cpu = cp.asnumpy(counts)  # Transfers results and waits for GPU.
elapsed = time.perf_counter() - start
print(f"CuPy time: {elapsed:.2f} seconds")

image = np.log1p(counts_cpu)

plt.figure(figsize=(6, 6))
plt.imshow(image, cmap="jet", origin="lower",
           extent=(-1.5, 1.5, -1.5, 1.5))
plt.colorbar()
plt.axis("equal")
plt.title(f"CuPy: {elapsed:.2f} s")
plt.savefig("julia_cupy.png", dpi=300)
plt.close()
