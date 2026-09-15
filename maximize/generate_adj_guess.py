import numpy as np
from mpi4py import MPI
import logging
import config
# Import functions defined in adjoint_dns.py
from adjoint_dns import run_adjoint_dns, advance_solver
from trans import pack_local_state, gather_to_zero

logging.getLogger('dedalus').setLevel(logging.WARNING)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        print(f"==================================================")
        print(f" Running Adjoint DNS at Ra={config.Ra}")
        print(f"==================================================")

    # 1. Specify the steady-state base flow file
    steady_file = f"steady_state_Ra_{config.Ra:.2e}.npy"

    if rank == 0:
        print(f" -> Base flow: {steady_file}")

    # 2. Create solver and initialize fields using create_adjoint_solver
    if rank == 0:
        print(" -> Initializing adjoint solver and loading base state...")
    solver, star_fields = run_adjoint_dns(steady_file)

    # 3. Advance the adjoint fields in time using advance_solver
    if rank == 0:
        print(" -> Advancing adjoint solver in time...")

    # Set duration and timestep as needed
    advance_solver(solver, duration=20.0, timestep=0.005)

    # 4. Pack local state coefficients and gather them to Rank 0
    local_x = pack_local_state(solver)
    global_x = gather_to_zero(local_x, comm)

    # 5. Save the final state to a file on Rank 0
    if rank == 0:
        output_filename = f"adjoint_star_Ra_{config.Ra:.2e}.npy"
        np.save(output_filename, global_x)
        print(f"==================================================")
        print(f" Adjoint DNS finished successfully!")
        print(f" Saved to '{output_filename}'")
        print(f"==================================================")

if __name__ == '__main__':
    main()      
