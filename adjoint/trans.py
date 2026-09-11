import numpy as np
from mpi4py import MPI

# ==============================================================================
# [1] Pack: Dedalus variables -> Local 1D array
# ==============================================================================
def pack_local_state(solver):
    local_arrays = []
    for field in solver.state:
        local_arrays.append(field['c'].flatten())
    # NumPy safely unifies real and complex into complex128
    return np.concatenate(local_arrays)

# ==============================================================================
# [2] Unpack: Local 1D array -> Dedalus variables
# ==============================================================================
def unpack_local_state(solver, local_array):
    offset = 0
    for field in solver.state:
        shape = field['c'].shape
        size = np.prod(shape)

        chunk = local_array[offset:offset+size].reshape(shape)
        if field['c'].dtype == np.float64:
            # Discard imaginary part only when converting back to real
            np.copyto(field['c'], chunk.real)
        else:
            np.copyto(field['c'], chunk)
        offset += size

# ==============================================================================
# [3] Gather: Local arrays from all cores -> Global array on Rank 0
# ==============================================================================
def gather_to_zero(local_array, comm):
    gathered_list = comm.gather(local_array, root=0)

    if comm.rank == 0:
        return np.concatenate(gathered_list)
    else:
        return None

# ==============================================================================
# [4] Scatter: Global array on Rank 0 -> Local arrays on all cores
# ==============================================================================
def scatter_from_zero(global_array, local_size, comm):
    sizes = comm.allgather(local_size)

    if comm.rank == 0:
        split_indices = np.cumsum(sizes)[:-1]
        chunks_list = np.split(global_array, split_indices)
    else:
        chunks_list = None

    local_array = comm.scatter(chunks_list, root=0)

    return local_array
