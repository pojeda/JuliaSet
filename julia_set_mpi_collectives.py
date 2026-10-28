from mpi4py import MPI
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
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = 3000
    max_iter = 300
    c = -0.7 + 0.27015j
    x = np.linspace(-1.5, 1.5, N)

    # Rank 0 divides the rows into blocks.
    if rank == 0:
        y = np.linspace(-1.5, 1.5, N)
        blocks = np.array_split(y, size)
    else:
        blocks = None

    comm.Barrier()
    start = MPI.Wtime()

    local_y = comm.scatter(blocks, root=0)
    local_image = julia(c, x, local_y, max_iter)
    results = comm.gather(local_image, root=0)

    if rank == 0:
        image = np.vstack(results)
        elapsed = MPI.Wtime() - start
        print(f"MPI collective time ({size} ranks): {elapsed:.2f} seconds")

        plt.imshow(np.log1p(image), extent=(-1.5, 1.5, -1.5, 1.5),
                   origin="lower", cmap="jet")
        plt.colorbar()
        plt.title(f"MPI collective ({size} ranks): {elapsed:.2f} s")
        plt.savefig("julia_mpi_collective.png", dpi=150)
        plt.close()
