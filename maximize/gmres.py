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

    # Build once to get array size
    dummy_solver, _, _ = run_dns(run_duration=0.0)
    local_size = len(pack_local_state(dummy_solver))
    del dummy_solver

    # =====================================================================
    # Worker phase (Rank > 0)
    # =====================================================================
    if rank > 0:
        while True:
            cmd = comm.bcast(None, root=0)
            if cmd == 'STOP':
                break

            elif cmd == 'EVAL':
                local_x = scatter_from_zero(None, local_size, comm)

                # Rebuild solver from scratch to reset past history
                solver, b, u = run_dns(run_duration=0.0)
                unpack_local_state(solver, local_x)
                advance_solver(solver, T_max, dt)
                new_local_x = pack_local_state(solver)
                gather_to_zero(new_local_x, comm)

                # Free memory
                del solver, b, u
                gc.collect()
        return

    # =====================================================================
    # Master phase (Rank 0)
    # =====================================================================
    else:
        print(f"==================================================")
        print(f" Starting Time-Stepper JFNK (Steady) for Ra = {config.Ra}")
        print(f" Using delta_T = {T_max}")
        print(f"==================================================")

        # 1. Load initial guess
        guess_file = config.INITIAL_GUESS_FILE
        try:
            x_k = np.load(guess_file)
            print(f"  Loaded state from: {guess_file}")
        except FileNotFoundError:
            print(f"  [ERROR] Could not find {guess_file}.")
            comm.bcast('STOP', root=0)
            return

        N_size = x_k.size

        # 2. Compute shooting residual
        def compute_shooting_residual(x_array):
            comm.bcast('EVAL', root=0)

            local_x = scatter_from_zero(x_array, local_size, comm)

            # Rank 0 joins the Dedalus computation
            solver, b, u = run_dns(run_duration=0.0)
            unpack_local_state(solver, local_x)

            advance_solver(solver, T_max, dt)
            new_local_x = pack_local_state(solver)

            new_global_x = gather_to_zero(new_local_x, comm)

            del solver, b, u
            gc.collect()

            return new_global_x - x_array

        # 3. Jacobian approximation using dynamic epsilon
        def apply_J_shooting(v_array, x_current, current_F):
            norm_x = np.linalg.norm(x_current)
            norm_v = np.linalg.norm(v_array)

            if norm_v < 1e-14:
                return np.zeros_like(v_array)
            epsilon = 1e-6 * (1.0 + norm_x) / norm_v
            F_plus = compute_shooting_residual(x_current + epsilon * v_array)

            return (F_plus - current_F) / epsilon

        # 4. Newton method main loop
        for i in range(config.MAX_ITER):
            current_F = compute_shooting_residual(x_k)
            b_array = -current_F

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

        # 5. Save results and finalize
        save_filename = f"steady_state_Ra_{config.Ra:.2e}.npy"
        np.save(save_filename, x_k)
        print(f"  Saved converged steady state to: {save_filename}")
        print(f"==================================================")

        comm.bcast('STOP', root=0)

if __name__ == "__main__":
    main()
