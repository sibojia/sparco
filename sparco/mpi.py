from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
procs = MPI.COMM_WORLD.Get_size()
root = 0

barrier = MPI.COMM_WORLD.Barrier
bcast = MPI.COMM_WORLD.Bcast
bcast_obj = MPI.COMM_WORLD.bcast
scatter = MPI.COMM_WORLD.Scatter
gather = MPI.COMM_WORLD.Gather
reduce = MPI.COMM_WORLD.Reduce

SUM = MPI.SUM
