import numpy as np
import dedalus.public as d3
from scipy.sparse.linalg import LinearOperator, gmres
import logging
import gc

# �~C��~B��~A�設�~Z
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# �~C~Q�~C��~C��~C��~B��~A�座�~Y系�~A�設�~Z
# ==========================================
Lx, Lz = 2, 1
Nx, Nz = 128, 64
Rayleigh = 10**4.1
Prandtl = 1
dealias = 3/2
dtype = np.float64

coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# ==========================================
# �~C~U�~B��~C��~C��~C~I�~A�解�~A~O�~V��~K�~O�~A��~Z義
# ==========================================
problem_fields = [
    dist.Field(name='p', bases=(xbasis,zbasis)),
    dist.Field(name='b', bases=(xbasis,zbasis)),
    dist.VectorField(coords, name='u', bases=(xbasis,zbasis)),
    dist.Field(name='tau_p'),
    dist.Field(name='tau_b1', bases=xbasis),
    dist.Field(name='tau_b2', bases=xbasis),
    dist.VectorField(coords, name='tau_u1', bases=xbasis),
    dist.VectorField(coords, name='tau_u2', bases=xbasis)
]
p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2 = problem_fields

kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1)
grad_b = d3.grad(b) + ez*lift(tau_b1)

# DNS�~A�解�~A~O�~V��~K�~O
problem = d3.IVP(problem_fields, namespace={**globals(), **locals()})
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)")
problem.add_equation("b(z=0) = Lz")
problem.add_equation("u(z=0) = 0")
problem.add_equation("b(z=Lz) = 0")
problem.add_equation("u(z=Lz) = 0")
problem.add_equation("integ(p) = 0")

# ==========================================
# �~E~M�~H~W�~I�~O~[�~A��~V��~U�
# ==========================================
# �~I��~P~F�~G~O�~A��~U�~V~K�~B�~U��~A��~E��~A��~A~_�~L�~H~W�~B~R�~@次�~E~C�~E~M�~H~W�~A��~U�~V~K
def pack_state(fields):
    return np.concatenate([f['c'].flatten() for f in fields])

# �~@次�~E~C�~E~M�~H~W�~A~K�~B~I�~E~C�~A��~L�~H~W�~A��~H��~A~Y
def unpack_state(x_array, fields):
    offset = 0
    for f in fields:
        size = f['c'].size
        f['c'][:] = x_array[offset : offset+size].reshape(f['c'].shape)
        offset += size
# ==========================================
# Nu�~A��~H�~W
# ==========================================
def compute_Nusselt():
    w = u @ ez
    vol_integral = d3.Integrate(w * b).evaluate()
    vol_average = vol_integral['g'][0, 0] / (Lx * Lz)

    Nu = 1.0 + vol_average / kappa
    return Nu

# ==========================================
# �~Z常解�~N�索
# ==========================================
# delta_T�~A| �~A~QDNS�~A��~Y~B�~V~S�~Y��~U�~A~U�~A~[�~A~_�~Z~[�~A��~C~U�~B��~C��~C��~C~I�~A��~@�
def run_dns_for_deltaT(x_array, delta_T):
    unpack_state(x_array, problem_fields)

    solver = problem.build_solver(d3.RK222)
    solver.sim_time = 0.0
    dt = 0.05 # 差�~H~F�~C��~B��~C~S�~B��~C��~A��~C~N�~B��~B��~B~R�~X��~A~P�~A~_�~B~A�~[��~Z�~B��~B��~C| �~B��~C~F�~C~C�~C~W

    while solver.sim_time < delta_T - 1e-8:
        step_dt = min(dt, delta_T - solver.sim_time)
        solver.step(step_dt)

    return pack_state(problem_fields)

# delta_T�~A| �~A~QDNS�~A��~Y~B�~V~S�~Y��~U�~A~U�~A~[�~A~_�~Z~[�~A��~C~U�~B��~C��~C��~C~I�~A��~I�~L~V�~G~O
def compute_shooting_residual(x_array, delta_T):
    x_final = run_dns_for_deltaT(x_array, delta_T)
#�~@~@�~@~@
    return x_final - x_array    # �~Y~B�~V~S delta_T �~@��~B~A�~A~_�~P�~^~\�~A~L�~@~A�~E~C�~A��~J��~E~K�~A��~P~L�~A~X�~A��~B~I�~Z常
�解

def apply_J_shooting(v_array, x_current, current_F, delta_T):
    """ GMR316227.7660168379ES�~A�渡�~A~Y 微�~O差�~H~F�~A��~B~H�~B~K�~C��~B��~C~S�~B��~C��~L�~H~W�~C��~C~Y�~B��~C~H�~C��~M """
    norm_x = np.linalg.norm(x_current)
    norm_v = np.linalg.norm(v_array)
    if norm_v < 1e-14:
        return np.zeros_like(v_array)

    epsilon = 1e-6 * (1.0 + norm_x) / norm_v
    F_plus = compute_shooting_residual(x_current + epsilon * v_array, delta_T)

    return (F_plus - current_F) / epsilon

# ==========================================
# 5. �~C��~B��~C��~C��~C��~C~W
# ==========================================
def solve_steady_shooting(initial_guess_file, delta_T=1.0):
    logger.info("==================================================")
    logger.info(f" Starting Time-Stepper JFNK (Steady) for Ra = {Rayleigh}")
    logger.info(f" Using delta_T = {delta_T}")
    logger.info("==================================================")

    try:
        x_k = np.load(initial_guess_file)
        logger.info(f"  Loaded state from: {initial_guess_file}")
    except FileNotFoundError:
        logger.error(f"  Could not find {initial_guess_file}.")
        raise

    N_size = x_k.size

    for i in range(15):
        # 1�~[~^�~A��~K差�~H�~W�~A��~A��~A~M1�~[~^DNS(delta_T�~H~F)�~A~L走�~B~K
        current_F = compute_shooting_residual(x_k, delta_T)
        b_array = -current_F

        residual_norm = np.linalg.norm(b_array) / np.sqrt(N_size)
        logger.info(f"  Newton Iteration {i}: RMS Residual = {residual_norm:.4e}")

        if residual_norm < 1e-14:
            logger.info("    -> Steady State Converged successfully!")
            break

        matvec = lambda v: apply_J_shooting(v, x_k, current_F, delta_T)
        J_op = LinearOperator((N_size, N_size), matvec=matvec)

        logger.info("    Solving inner GMRES...")
        # 1�~[~^�~A�GMRES�~B��~C~F�~C~C�~C~W�~A��~A��~A~M1�~[~^DNS�~A~L走�~B~K
        delta_x, exit_code = gmres(J_op, b_array, rtol=1e-2, restart=20, maxiter=5)

        x_k = x_k + delta_x
        gc.collect()

    save_filename = f"state_Ra{Rayleigh}.npy"
    np.save(save_filename, x_k)
    logger.info(f"  Saved converged steady state to: {save_filename}")

    # Nu�~A��~H�~W�~A��~G��~J~[
    unpack_state(x_k, problem_fields)
    Nu_val = compute_Nusselt()
    logger.info("==================================================")
    logger.info(f"  => Final Nusselt Number (Nu) = {Nu_val:.6f}")
    logger.info("==================================================")

    return save_filename

if __name__ == "__main__":
    # delta_T �~A��~Y~B�~V~S�~A| �~A~QDNS�~B~R�~L�~A~F
    solve_steady_shooting("state_Ra10000.npy", delta_T=100)
    pass
