# JuliaSet

#ml GCCcore/11.2.0 Python/3.9.6
ml GCC/15.2.0 OpenMPI/5.0.10  Python/3.14.2
ml CUDA/12.9.1

python -m venv vpyenv-python-course

source vpyenv-python-course/bin/activate  

pip install mpi4py "numpy==2.4.6"  matplotlib numba numba-cuda cupy-cuda12x  --cache-dir /proj/nobackup/po_hpc/.pip

python julia_set_serial.py

mpirun -np 10 julia_set_mpi_p2p.py

mpirun -np 10 julia_set_mpi_collectives.py
