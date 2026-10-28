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
    rank, size = comm.Get_rank(), comm.Get_size()
    N = 3000
    max_iter = 300
    c = -0.7 + 0.27015j
    x = np.linspace(-1.5, 1.5, N)
    y = np.linspace(-1.5, 1.5, N)

    # Each rank computes a different group of rows.
    first = rank * N // size
    last = (rank + 1) * N // size

    comm.Barrier()
    start = MPI.Wtime()
    local = julia(c, x, y[first:last], max_iter)

    if rank == 0:
        image = np.empty((N, N), dtype=np.int32)
        image[first:last] = local
        for r in range(1, size):
            a, b = r * N // size, (r + 1) * N // size
            comm.Recv(image[a:b], source=r)

        elapsed = MPI.Wtime() - start
        print(f"MPI time ({size} ranks): {elapsed:.2f} seconds")
        plt.imshow(np.log1p(image), extent=(-1.5, 1.5, -1.5, 1.5),
                   origin="lower", cmap="jet")
        plt.colorbar()
        plt.title(f"MPI ({size} ranks): {elapsed:.2f} s")
        plt.savefig("julia_mpi.png", dpi=150)
        plt.close()
    else:
        comm.Send(local, dest=0)
