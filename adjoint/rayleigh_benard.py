import numpy as np
import dedalus.public as d3
import logging
import os
logger = logging.getLogger(__name__)

import config

def run_dns(run_duration=config.T_max, timestep=config.dt, Rayleigh=config.Ra, Prandtl=config.Pr, Lx=config.Lx, Lz=config.Lz, Nx=config.Nx, Nz=config.Nz, restart_file=None):

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
    problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)")
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
    #finally:
    #    solver.log_stats()

    finally:
        try:
            solver.log_stats()
        except AttributeError:
            pass

    return solver, b, u


#if __name__ == '__main__':
#    solver, final_b, final_u = run_dns(run_duration=10.0, timestep=0.05, Rayleigh=1e4)
