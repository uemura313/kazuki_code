import numpy as np
from mpi4py import MPI

# ==============================================================================
# [1] Pack: Dedalus変数 -> ローカル1次元配列
# ==============================================================================
def pack_local_state(solver):
    local_arrays = []
    for field in solver.state:
        local_arrays.append(field['c'].flatten())
    # NumPyの仕様で、実数と複素数が混ざっても全て complex128 に安全に統一されます
    return np.concatenate(local_arrays)

# ==============================================================================
# [2] Unpack: ローカル1次元配列 -> Dedalus変数
# ==============================================================================
def unpack_local_state(solver, local_array):
    offset = 0
    for field in solver.state:
        shape = field['c'].shape
        size = np.prod(shape)

        chunk = local_array[offset:offset+size].reshape(shape)
        if field['c'].dtype == np.float64:
            # 複素数から実数へ戻す時だけ、虚部を切り捨てる
            np.copyto(field['c'], chunk.real)
        else:
            np.copyto(field['c'], chunk)
        offset += size

# ==============================================================================
# [3] Gather: 全コアのローカル配列 -> Rank 0のグローバル配列
# ==============================================================================
def gather_to_zero(local_array, comm):
    # 小文字の gather は、各コアの配列をそのままRank 0に集めて「リスト」にしてくれます
    gathered_list = comm.gather(local_array, root=0)

    if comm.rank == 0:
        # Rank 0 は、受け取ったリストを1本のNumPy配列に結合するだけ
        return np.concatenate(gathered_list)
    else:
        return None

# ==============================================================================
# [4] Scatter: Rank 0のグローバル配列 -> 全コアのローカル配列
# ==============================================================================
def scatter_from_zero(global_array, local_size, comm):
    # まず、各コアがいくつの要素を欲しがっているかを集める
    sizes = comm.allgather(local_size)

    if comm.rank == 0:
        # Rank 0 は、長い配列を sizes に従って切り分け、リストに詰める
        split_indices = np.cumsum(sizes)[:-1]
        chunks_list = np.split(global_array, split_indices)
    else:
        chunks_list = None

    # 小文字の scatter は、リストの中身を各コアに安全に配ってくれます
    local_array = comm.scatter(chunks_list, root=0)

    return local_array
