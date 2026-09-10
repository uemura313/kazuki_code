import numpy as np
from mpi4py import MPI
import logging
import gc
from scipy.sparse.linalg import LinearOperator, gmres
import config

from adjoint_dns import create_adjoint_solver, advance_solver
from trans import pack_local_state, unpack_local_state, gather_to_zero, scatter_from_zero

logging.getLogger('dedalus').setLevel(logging.WARNING)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    T_MAP = config.T_max
    DT = config.dt
    BASE_FILE = config.INITIAL_GUESS_FILE

    # 1. ソルバーの構築（使い回し）
    solver, star_fields = create_adjoint_solver(BASE_FILE)
    local_x = pack_local_state(solver)
    local_size = len(local_x)

    # 2. 全未知数の総数 (N_size) を全コアで共有
    global_x0 = gather_to_zero(local_x, comm)
    N_size = global_x0.size if rank == 0 else 0
    N_size = comm.bcast(N_size, root=0)

    # 探索ベクトル（全コアで常に同じ値を保持）
    x_k = np.zeros(N_size)

    if rank == 0:
        print(f"==================================================")
        print(f" Solving Symmetrical Parallel JFNK for Ra = {config.Ra}")
        print(f"==================================================")

    # 3. 残差評価関数（全コアが一斉に実行して同期する）
    def compute_shooting_residual(x_array):
        local_xi = scatter_from_zero(x_array, local_size, comm)
        unpack_local_state(solver, local_xi)

        advance_solver(solver, T_MAP, DT)

        new_local_x = pack_local_state(solver)
        new_global_x = gather_to_zero(new_local_x, comm)

        # Rank 0 で結合された配列を、全コアにブロードキャスト（共有）
        new_global_x = comm.bcast(new_global_x, root=0)
        return new_global_x - x_array

    def apply_J_shooting(v_array, x_current, current_F):
        norm_x = np.linalg.norm(x_current)
        norm_v = np.linalg.norm(v_array)
        if norm_v < 1e-14:
            return np.zeros_like(v_array)

        epsilon = 1e-6 * (1.0 + norm_x) / norm_v
        F_plus = compute_shooting_residual(x_current + epsilon * v_array)
        return (F_plus - current_F) / epsilon

    # 4. ニュートンループ（全コアが同時に同じ計算ルートを通る）
    for i in range(config.MAX_ITER):
        current_F = compute_shooting_residual(x_k)
        b_array = -current_F
        residual_norm = np.linalg.norm(b_array) / np.sqrt(N_size)

        if rank == 0:
            print(f"  Newton Iteration {i}: RMS Residual = {residual_norm:.4e}")

        if residual_norm < 1e-14:
            if rank == 0:
                print("    -> Adjoint State Converged successfully!")
            break

        # 全コアがLinearOperatorを構築し、全コアがGMRESを実行することで、
        # 内部のapply_J_shootingが1コアも欠けることなく完全に同期して呼ばれます
        matvec = lambda v: apply_J_shooting(v, x_k, current_F)
        J_op = LinearOperator((N_size, N_size), matvec=matvec)

        if rank == 0:
            print("    Solving inner GMRES...")

        delta_x, exit_code = gmres(J_op, b_array, rtol=1e-2, restart=20, maxiter=5)
        x_k = x_k + delta_x

    if rank == 0:
        save_filename = f"adjoint_star_Ra_{config.Ra:.2e}.npy"
        np.save(save_filename, x_k)
        print(f"  Saved adjoint variables to: {save_filename}")
        print(f"==================================================")

if __name__ == "__main__":
    main()
