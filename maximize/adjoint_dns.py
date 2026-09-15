import numpy as np
import dedalus.public as d3
import os
from mpi4py import MPI
import config
from trans import scatter_from_zero

def run_adjoint_dns(base_state_file, alpha_file="current_alpha.npy"):
    Lx, Lz = config.Lx, config.Lz
    Nx, Nz = config.Nx, config.Nz
    Rayleigh = config.Ra
    Prandtl = config.Pr

    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=3/2)
    zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=3/2)

    # 1. Base Flow Setup
    p_base = dist.Field(name='p_base', bases=(xbasis,zbasis))
    b_base = dist.Field(name='b_base', bases=(xbasis,zbasis))
    u_base = dist.VectorField(coords, name='u_base', bases=(xbasis,zbasis))
    tau_p_b = dist.Field(name='tau_p_b')
    tau_b1_b = dist.Field(name='tau_b1_b', bases=xbasis)
    tau_b2_b = dist.Field(name='tau_b2_b', bases=xbasis)
    tau_u1_b = dist.VectorField(coords, name='tau_u1_b', bases=xbasis)
    tau_u2_b = dist.VectorField(coords, name='tau_u2_b', bases=xbasis)

    base_fields = [p_base, b_base, u_base, tau_p_b, tau_b1_b, tau_b2_b, tau_u1_b, tau_u2_b]

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        global_base = np.load(base_state_file)
    else:
        global_base = None

    offset = 0
    for f in base_fields:
        local_size = f['c'].size
        global_size = comm.allreduce(local_size, op=MPI.SUM)

        if rank == 0:
            field_global_data = global_base[offset : offset + global_size]
            offset += global_size
        else:
            field_global_data = None

        field_local_data = scatter_from_zero(field_global_data, local_size, comm)
        chunk = field_local_data.reshape(f['c'].shape)
        if f['c'].dtype == np.float64:
            np.copyto(f['c'], chunk.real)
        else:
            np.copyto(f['c'], chunk)

    # 2. Adjoint (Star) Variables & Alpha Setup
    p_star = dist.Field(name='p_star', bases=(xbasis,zbasis))
    b_star = dist.Field(name='b_star', bases=(xbasis,zbasis))
    u_star = dist.VectorField(coords, name='u_star', bases=(xbasis,zbasis))
    tau_p_s = dist.Field(name='tau_p_s')
    tau_b1_s = dist.Field(name='tau_b1_s', bases=xbasis)
    tau_b2_s = dist.Field(name='tau_b2_s', bases=xbasis)
    tau_u1_s = dist.VectorField(coords, name='tau_u1_s', bases=xbasis)
    tau_u2_s = dist.VectorField(coords, name='tau_u2_s', bases=xbasis)

    star_fields = [p_star, b_star, u_star, tau_p_s, tau_b1_s, tau_b2_s, tau_u1_s, tau_u2_s]

    alpha = dist.Field(name='alpha', bases=(xbasis, zbasis))

    # 【対応部分】安全な alpha のロード（ファイルがなければ g 空間で 1.0 に設定）
    file_exists = False
    global_alpha = None
    if rank == 0:
        if alpha_file and os.path.exists(alpha_file):
            try:
                global_alpha = np.load(alpha_file)
                file_exists = True
            except Exception:
                file_exists = False

    file_exists = comm.bcast(file_exists, root=0)

    if file_exists:
        local_size_alpha = alpha['c'].size
        alpha_local = scatter_from_zero(global_alpha, local_size_alpha, comm)
        np.copyto(alpha['c'], alpha_local.reshape(alpha['c'].shape).real)
    else:
        alpha['g'] = 1.0

    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)
    ex, ez = coords.unit_vector_fields(dist)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)

    grad_u_star = d3.grad(u_star) + ez*lift(tau_u1_s)
    grad_b_star = d3.grad(b_star) + ez*lift(tau_b1_s)

    grad_u_base = d3.grad(u_base)
    grad_b_base = d3.grad(b_base)
    w_base = u_base @ ez
    w_star = u_star @ ez

    problem = d3.IVP(star_fields, namespace=locals())

    problem.add_equation("trace(grad_u_star) + tau_p_s = 0")
    problem.add_equation("dt(b_star) - kappa*div(grad_b_star) + lift(tau_b2_s) = -(u_base @ grad_b_star) - w_base + w_star")
    problem.add_equation("dt(u_star) - nu*div(grad_u_star) + grad(p_star) + lift(tau_u2_s) = -(u_base @ grad_u_star) - (grad_u_base @ u_star) - b_base*ez - b_star*grad_b_base - (alpha**2)*u_star")

    problem.add_equation("b_star(z=0) = 0")
    problem.add_equation("u_star(z=0) = 0")
    problem.add_equation("b_star(z=Lz) = 0")
    problem.add_equation("u_star(z=Lz) = 0")
    problem.add_equation("integ(p_star) = 0")

    solver = problem.build_solver(d3.RK222)
    return solver, star_fields

def advance_solver(solver, duration, timestep, analysis_interval=1.0):
    """
    Advance the solver in time with progress monitoring and data output.
    """
    solver.sim_time = 0.0
    solver.stop_sim_time = duration

    # Optional: Add file handler to save data for visualization and analysis
    analysis = solver.evaluator.add_file_handler('adjoint_snapshots', sim_dt=analysis_interval, max_writes=500)
    analysis.add_tasks(solver.state) # Saves all state fields (u_star, b_star, p_star, etc.)

    # Main time-integration loop
    print("Starting time integration...")
    while solver.proceed:
        solver.step(timestep)

        # Print progress every 100 steps (or based on iteration)
        if solver.iteration % 100 == 0:
            print(f"Iteration: {solver.iteration}, Sim Time: {solver.sim_time:.3f} / {duration:.3f}")

    print("Time integration finished.")
