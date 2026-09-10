import numpy as np
import dedalus.public as d3
from mpi4py import MPI
import config
from trans import scatter_from_zero

def create_adjoint_solver(base_state_file):
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
    if comm.rank == 0:
        global_base = np.load(base_state_file)
    else:
        global_base = None

    from trans import scatter_from_zero
    offset = 0

    for f in base_fields:
        local_size = f['c'].size
        # 全コアの local_size を合計して、このフィールドのグローバルサイズを計算
        global_size = comm.allreduce(local_size, op=MPI.SUM)

        # Rank 0 が配列の中から「このフィールドの分」だけを正確に切り出す
        if comm.rank == 0:
            field_global_data = global_base[offset : offset + global_size]
            offset += global_size
        else:
            field_global_data = None

        # フィールド単体の配列を、各コアに安全に分配
        field_local_data = scatter_from_zero(field_global_data, local_size, comm)

        # 自分のフィールドにデータを流し込む
        chunk = field_local_data.reshape(f['c'].shape)
        if f['c'].dtype == np.float64:
            np.copyto(f['c'], chunk.real)
        else:
            np.copyto(f['c'], chunk)
    # 2. Adjoint (Star) Variables Setup
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
    # 分散次元の結合を避けるため、(alpha**2)*u_star は右辺に配置
    problem.add_equation("dt(u_star) - nu*div(grad_u_star) + grad(p_star) + lift(tau_u2_s) = -(u_base @ grad_u_star) - (grad_u_base @ u_star) - b_base*ez - b_star*grad_b_base - (alpha**2)*u_star")

    problem.add_equation("b_star(z=0) = 0")
    problem.add_equation("u_star(z=0) = 0")
    problem.add_equation("b_star(z=Lz) = 0")
    problem.add_equation("u_star(z=Lz) = 0")
    problem.add_equation("integ(p_star) = 0")

    solver = problem.build_solver(d3.RK222)
    return solver, star_fields

def advance_solver(solver, duration, timestep):
    solver.sim_time = 0.0
    solver.stop_sim_time = duration
    while solver.proceed:
        solver.step(timestep)
