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
    solver.stop_sim_time = solver.sim_time + duration
    while solver.proceed:
        solver.step(timestep)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    T_max = config.T_max
    dt = config.dt

    # 初期設定のため1度だけ構築してサイズを取得
    dummy_solver, _, _ = run_dns(run_duration=0.0)
    local_size = len(pack_local_state(dummy_solver))
    del dummy_solver

    # =====================================================================
    # ワーカーフェーズ (Rank 1, 2, 3)
    # =====================================================================
    if rank > 0:
        while True:
            cmd = comm.bcast(None, root=0)
            if cmd == 'STOP':
                break

            elif cmd == 'EVAL':
                local_x = scatter_from_zero(None, local_size, comm)

                # 【最重要】毎回ゼロからソルバーを構築し、過去の履歴を完全にリセットする
                solver, b, u = run_dns(run_duration=0.0)
                unpack_local_state(solver, local_x)

                advance_solver(solver, T_max, dt)

                new_local_x = pack_local_state(solver)
                gather_to_zero(new_local_x, comm)

                # メモリ解放
                del solver, b, u
                gc.collect()
        return

    # =====================================================================
    # マスターフェーズ (Rank 0) - 自作の完璧なJFNKアルゴリズム
    # =====================================================================
    else:
        print(f"==================================================")
        print(f" Starting Time-Stepper JFNK (Steady) for Ra = {config.Ra}")
        print(f" Using delta_T = {T_max}")
        print(f"==================================================")

        # 1. 初期推測値の読み込み
        guess_file = config.INITIAL_GUESS_FILE
        try:
            x_k = np.load(guess_file)
            print(f"  Loaded state from: {guess_file}")
        except FileNotFoundError:
            print(f"  [ERROR] Could not find {guess_file}.")
            comm.bcast('STOP', root=0)
            return

        N_size = x_k.size

        # 2. 写像評価関数（ワーカーに指示を出して F(x) - x を計算）
        def compute_shooting_residual(x_array):
            # ワーカーに「評価開始」を合図
            comm.bcast('EVAL', root=0)

            # 1. 分割した配列を受け取る（Rank 0自身も）
            local_x = scatter_from_zero(x_array, local_size, comm)

            # 2. 【最重要】Rank 0自身もDedalusの計算に合流する！
            solver, b, u = run_dns(run_duration=0.0)
            unpack_local_state(solver, local_x)

            # 時間積分（ここでRank 0〜3が裏で激しく通信しながら計算を進めます）
            advance_solver(solver, T_max, dt)

            new_local_x = pack_local_state(solver)

            # 3. 全コアの結果を Rank 0 に集約
            new_global_x = gather_to_zero(new_local_x, comm)

            # メモリ解放（使い回し防止）
            del solver, b, u
            gc.collect()

            return new_global_x - x_array
        # 3. 動的 epsilon によるヤコビアン近似（以前の成功コードをそのまま使用）
        def apply_J_shooting(v_array, x_current, current_F):
            norm_x = np.linalg.norm(x_current)
            norm_v = np.linalg.norm(v_array)

            if norm_v < 1e-14:
                return np.zeros_like(v_array)

            epsilon = 1e-6 * (1.0 + norm_x) / norm_v
            F_plus = compute_shooting_residual(x_current + epsilon * v_array)

            return (F_plus - current_F) / epsilon

        # 4. ニュートン法のメインループ
        for i in range(config.MAX_ITER):
            current_F = compute_shooting_residual(x_k)
            b_array = -current_F

            # RMSノルムで正確な誤差を評価
            residual_norm = np.linalg.norm(b_array) / np.sqrt(N_size)
            print(f"  Newton Iteration {i}: RMS Residual = {residual_norm:.4e}")

            if residual_norm < 1e-14:
                print("    -> Steady State Converged successfully!")
                break

            matvec = lambda v: apply_J_shooting(v, x_k, current_F)
            J_op = LinearOperator((N_size, N_size), matvec=matvec)

            print("    Solving inner GMRES...")
            delta_x, exit_code = gmres(J_op, b_array, rtol=1e-2, restart=20, maxiter=5)

            x_k = x_k + delta_x
            gc.collect()

        # 5. 収束結果の保存と終了処理
        save_filename = f"steady_state_Ra_{config.Ra:.2e}.npy"
        np.save(save_filename, x_k)
        print(f"  Saved converged steady state to: {save_filename}")
        print(f"==================================================")

        comm.bcast('STOP', root=0)

if __name__ == "__main__":
    main()
