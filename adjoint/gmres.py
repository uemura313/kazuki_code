import numpy as np
from mpi4py import MPI
import logging
import gc
from scipy.sparse.linalg import LinearOperator, gmres
import config

from rayleigh_benard import run_dns
from trans import pack_local_state, unpack_local_state, gather_to_zero, scatter_from_zero

logging.getLogger('dedalus').setLevel(logging.WARNING)

def advance_solver(solver, duration, timestep):
    solver.sim_time = 0.0
    solver.stop_sim_time = duration
    while solver.proceed:
        solver.step(timestep)

def load_initial_guess_safely(filename, solver, comm):
    """
    コア数に依存せず、.npyファイルから各フィールドに正しくデータを流し込み、
    現在の並列数に合わせたローカル配列 (local_x) を生成する安全な関数
    """
    if comm.rank == 0:
        try:
            global_base = np.load(filename)
            print(f"  Loaded state safely from: {filename}")
        except FileNotFoundError:
            print(f"  [ERROR] Could not find {filename}.")
            return None
    else:
        global_base = None

    offset = 0
    for field in solver.state:
        local_size = field['c'].size
        global_size = comm.allreduce(local_size, op=MPI.SUM)
        
        if comm.rank == 0:
            field_global_data = global_base[offset : offset + global_size]
            offset += global_size
        else:
            field_global_data = None
            
        field_local_data = scatter_from_zero(field_global_data, local_size, comm)
        
        chunk = field_local_data.reshape(field['c'].shape)
        if field['c'].dtype == np.float64:
            np.copyto(field['c'], chunk.real)
        else:
            np.copyto(field['c'], chunk)

    # フィールドに正しく入った状態から、現在の並列環境用の配列を作る
    return pack_local_state(solver)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    T_max = config.T_max
    dt = config.dt

    # 1度だけソルバーを構築して使い回す（デッドロック回避）
    solver, b, u = run_dns(run_duration=0.0)
    
    if rank == 0:
        print(f"==================================================")
        print(f" Starting Symmetrical Parallel JFNK for Ra = {config.Ra}")
        print(f" Using delta_T = {T_max}")
        print(f"==================================================")

    # コア数に依存しない安全な読み込み
    guess_file = config.INITIAL_GUESS_FILE
    local_x = load_initial_guess_safely(guess_file, solver, comm)
    
    if local_x is None:
        return # ファイルがなければ終了

    local_size = len(local_x)
    global_x0 = gather_to_zero(local_x, comm)
    N_size = global_x0.size if rank == 0 else 0
    N_size = comm.bcast(N_size, root=0)

    # 全コアが共有する探索ベクトル
    x_k = np.zeros(N_size)
    if rank == 0:
        x_k[:] = global_x0

    x_k = comm.bcast(x_k, root=0)

    # 写像評価関数（全コアが一斉に実行）
    def compute_shooting_residual(x_array):
        local_xi = scatter_from_zero(x_array, local_size, comm)
        unpack_local_state(solver, local_xi)
        
        advance_solver(solver, T_max, dt)
        
        new_local_x = pack_local_state(solver)
        new_global_x = gather_to_zero(new_local_x, comm)
        new_global_x = comm.bcast(new_global_x, root=0)
        
        return new_global_x - x_array

    # ヤコビアン近似（全コアが一斉に実行）
    def apply_J_shooting(v_array, x_current, current_F):
        norm_x = np.linalg.norm(x_current)
        norm_v = np.linalg.norm(v_array)
        if norm_v < 1e-14:
            return np.zeros_like(v_array)

        epsilon = 1e-6 * (1.0 + norm_x) / norm_v
        F_plus = compute_shooting_residual(x_current + epsilon * v_array)
        return (F_plus - current_F) / epsilon

    # ニュートン法のメインループ
    for i in range(config.MAX_ITER):
        current_F = compute_shooting_residual(x_k)
        b_array = -current_F
        residual_norm = np.linalg.norm(b_array) / np.sqrt(N_size)
        
        if rank == 0:
            print(f"  Newton Iteration {i}: RMS Residual = {residual_norm:.4e}")

        if residual_norm < 1e-14:
            if rank == 0:
                print("    -> Steady State Converged successfully!")
            break

        matvec = lambda v: apply_J_shooting(v, x_k, current_F)
        J_op = LinearOperator((N_size, N_size), matvec=matvec)
        
        if rank == 0:
            print("    Solving inner GMRES...")
            
        delta_x, exit_code = gmres(J_op, b_array, rtol=1e-2, restart=20, maxiter=5)
        x_k = x_k + delta_x

    # 収束結果の保存（保存前に、フィールド単位の綺麗な並びに直して保存する）
    if rank == 0:
        save_filename = f"steady_state_Ra_{config.Ra:.2e}.npy"
        
        # 最終状態を全フィールドに反映
    local_xi = scatter_from_zero(x_k, local_size, comm)
    unpack_local_state(solver, local_xi)
    
    # 物理場から「変数ごと」の正しい順番で配列を取り直して保存
    if rank == 0:
        final_global_array = []
    
    for field in solver.state:
        local_data = field['c'].flatten()
        gathered_list = comm.gather(local_data, root=0)
        if rank == 0:
            final_global_array.append(np.concatenate(gathered_list))
            
    if rank == 0:
        final_save_array = np.concatenate(final_global_array)
        np.save(save_filename, final_save_array)
        print(f"  Saved converged steady state to: {save_filename}")
        print(f"==================================================")

if __name__ == "__main__":
    main()
