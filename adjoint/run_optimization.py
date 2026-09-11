import mpi4py.MPI as MPI
import config
import gmres
import adjoint_newton
from update_alpha import update_alpha_step

def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    max_opt_iters = 50
    convergence_tolerance = 1e-14
    step_size = 0.1

    for opt_iter in range(max_opt_iters):
        if rank == 0:
            print(f"\n==================================================")
            print(f" >>> Topology Optimization Iteration: {opt_iter} <<<")
            print(f"==================================================")

        # 1. solve the base variables
        gmres.main()

        # 2. solve the adjoint variables
        config.INITIAL_GUESS_FILE = f"steady_state_Ra_{config.Ra:.2e}.npy"
        adjoint_newton.main()

        # 3. evaluate the sensitibity and residual
        max_sens = update_alpha_step(step_size=step_size)

        # 4. convergence criterion
        if max_sens < convergence_tolerance:
            if rank == 0:
                print(f"\n[SUCCESS] Optimization converged at iteration {opt_iter}!")
            break

if __name__ == "__main__":
    main()
