import time
import numpy as np
from numba import cuda
import matplotlib.pyplot as plt


@cuda.jit
def julia(counts, N, max_iter, cr, ci):
    x, y = cuda.grid(2)
    if x >= N or y >= N:
        return

    zr = -1.5 + 3.0 * x / (N - 1)
    zi = -1.5 + 3.0 * y / (N - 1)
    count = 0

    for _ in range(max_iter):
        next_zr = zr * zr - zi * zi + cr
        zi = 2.0 * zr * zi + ci
        zr = next_zr
        count += 1
        if zr * zr + zi * zi > 4.0:
            break

    counts[y, x] = count


if __name__ == "__main__":
    N = 5000
    max_iter = 300
    cr, ci = -0.7, 0.27015
    counts_gpu = cuda.device_array((N, N), dtype=np.int32)
    threads = (16, 16)
    blocks = ((N + 15) // 16, (N + 15) // 16)

    # Compile the kernel once before measuring its execution time.
    julia[(1, 1), (1, 1)](counts_gpu, N, max_iter, cr, ci)
    cuda.synchronize()

    start = time.perf_counter()
    julia[blocks, threads](counts_gpu, N, max_iter, cr, ci)
    counts = counts_gpu.copy_to_host()  # Also waits for the GPU to finish.
    elapsed = time.perf_counter() - start
    print(f"Numba CUDA time: {elapsed:.2f} seconds")

    image = np.log1p(counts)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="jet", origin="lower",
               extent=(-1.5, 1.5, -1.5, 1.5))
    plt.colorbar()
    plt.axis("equal")
    plt.title(f"Numba CUDA: {elapsed:.2f} s")
    plt.savefig("julia_numba.png", dpi=300)
    plt.close()
