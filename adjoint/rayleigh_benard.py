import numpy as np
import dedalus.public as d3
import logging
import os
from mpi4py import MPI
logger = logging.getLogger(__name__)

import config
from trans import scatter_from_zero

def run_dns(run_duration=config.T_max, timestep=config.dt, Rayleigh=config.Ra, Prandtl=config.Pr, Lx=config.Lx, Lz=config.Lz, Nx=config.Nx, Nz=config.Nz, restart_file=None, alpha_file="current_alpha.npy"):

    comm = MPI.COMM_WORLD
    rank = comm.rank

    # Internal numerical parameters
    dealias = 3/2
    timestepper = d3.RK222
    dtype = np.float64

    # Coordinates
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=dtype)
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
    zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

    # Fields variables
    p = dist.Field(name='p', bases=(xbasis,zbasis))
    b = dist.Field(name='b', bases=(xbasis,zbasis))
    u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
    tau_p = dist.Field(name='tau_p')
    tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
    tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

# Alpha field for topology optimization
    alpha = dist.Field(name='alpha', bases=(xbasis, zbasis))

    # Load alpha safely
    file_exists = False
    global_alpha = None
    if rank == 0:
        if alpha_file and os.path.exists(alpha_file):
            try:
                global_alpha = np.load(alpha_file)
                file_exists = True
                logger.info(f"Loaded alpha from {alpha_file}")
            except Exception:
                file_exists = False

    file_exists = comm.bcast(file_exists, root=0)

    if file_exists:
        # ファイルがある場合は従来通り c 空間の係数としてロード
        local_size_alpha = alpha['c'].size
        alpha_local = scatter_from_zero(global_alpha, local_size_alpha, comm)
        np.copyto(alpha['c'], alpha_local.reshape(alpha['c'].shape).real)
    else:
        # ファイルがない初期状態では、物理グリッド (g空間) で一様に 1.0 を設定する
        if rank == 0:
            logger.info(f"No {alpha_file} found. Initializing alpha = 1.0 in grid space.")
        alpha['g'] = 1.0

    # Substitutions
    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)
    x, z = dist.local_grids(xbasis, zbasis)
    ex, ez = coords.unit_vector_fields(dist)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    grad_u = d3.grad(u) + ez*lift(tau_u1)
    grad_b = d3.grad(b) + ez*lift(tau_b1)

    # Problem
    problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
    problem.add_equation("trace(grad_u) + tau_p = 0")
    problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
    # 【ペナルティ項の追加】右辺に -(alpha**2)*u を付与
    problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u) - (alpha**2)*u")
    
    problem.add_equation("b(z=0) = Lz")
    problem.add_equation("u(z=0) = 0")
    problem.add_equation("b(z=Lz) = 0")
    problem.add_equation("u(z=Lz) = 0")
    problem.add_equation("integ(p) = 0")

    # Solver
    solver = problem.build_solver(timestepper)

    # -----------------------------------------------------------------------------------------------
    # Initial conditions
    # -----------------------------------------------------------------------------------------------
    if restart_file is not None and os.path.exists(restart_file):
        logger.info(f"=== Loading state from {restart_file} and resuming ===")
        solver.load_state(restart_file)
    else:
        logger.info("=== Starting from conduction solution with perturbation ===")
        b.fill_random('g', seed=42, distribution='normal', scale=1e-3)
        b['g'] *= z * (Lz - z)
        b['g'] += Lz - z

    # calculation time set
    solver.stop_sim_time = solver.sim_time + run_duration

    # -----------------------------------------------------------------------------------------------
    # Data conservation
    # -----------------------------------------------------------------------------------------------
    checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt=run_duration, max_writes=1)
    checkpoints.add_tasks(solver.state)

    # -----------------------------------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------------------------------
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            solver.step(timestep)
            if (solver.iteration-1) % 10 == 0:
                logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise

    finally:
        try:
            solver.log_stats()
        except AttributeError:
            pass

    return solver, b, u
