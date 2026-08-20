import numpy as np
import dedalus.public as d3
from scipy.sparse.linalg import LinearOperator, gmres
import logging
import gc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# Parameters and Coordinates
# ==========================================
Lx, Lz = 2.0, 1.0
Nx, Nz = 128, 64
Rayleigh = 10**4.1
Prandtl = 1.0
dealias = 3/2
dtype = np.float64

coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# ==========================================
# Fields and Equations
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

# DNS IVP problem setup
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
# State Vector Pack/Unpack Functions
# ==========================================
# Pack fields into a 1D array
def pack_state(fields):
    return np.concatenate([f['c'].flatten() for f in fields])

# Unpack a 1D array back into fields
def unpack_state(x_array, fields):
    offset = 0
    for f in fields:
        size = f['c'].size
        f['c'][:] = x_array[offset : offset+size].reshape(f['c'].shape)
        offset += size

# ==========================================
# Compute Nusselt Number
# ==========================================
def compute_Nusselt():
    w = u @ ez
    vol_integral = d3.Integrate(w * b).evaluate()
    vol_average = vol_integral['g'][0, 0] / (Lx * Lz)

    Nu = 1.0 + vol_average / kappa
    return Nu

# ==========================================
# Steady State Search (Shooting Method)
# ==========================================
# Run DNS for a period of delta_T
def run_dns_for_deltaT(x_array, delta_T):
    unpack_state(x_array, problem_fields)

    solver = problem.build_solver(d3.RK222)
    solver.sim_time = 0.0
    dt = 0.05  # Fixed time step

    while solver.sim_time < delta_T - 1e-8:
        step_dt = min(dt, delta_T - solver.sim_time)
        solver.step(step_dt)

    return pack_state(problem_fields)

# Compute shooting residual: F(x) = x_final - x_initial
def compute_shooting_residual(x_array, delta_T):
    x_final = run_dns_for_deltaT(x_array, delta_T)
    return x_final - x_array    

# Jacobian-vector product using finite difference for GMRES
def apply_J_shooting(v_array, x_current, current_F, delta_T):
    norm_x = np.linalg.norm(x_current)
    norm_v = np.linalg.norm(v_array)
    
    if norm_v < 1e-14:
        return np.zeros_like(v_array)

    epsilon = 1e-6 * (1.0 + norm_x) / norm_v
    F_plus = compute_shooting_residual(x_current + epsilon * v_array, delta_T)

    return (F_plus - current_F) / epsilon

# ==========================================
# Main Newton-Krylov Solver
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
        # 1. Run DNS to evaluate residual
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
        # 2. Inner GMRES loop
        delta_x, exit_code = gmres(J_op, b_array, rtol=1e-2, restart=20, maxiter=5)

        x_k = x_k + delta_x
        gc.collect()

    save_filename = f"state_Ra{Rayleigh}.npy"
    np.save(save_filename, x_k)
    logger.info(f"  Saved converged steady state to: {save_filename}")

    # Evaluate Final Nusselt number
    unpack_state(x_k, problem_fields)
    Nu_val = compute_Nusselt()
    logger.info("==================================================")
    logger.info(f"  => Final Nusselt Number (Nu) = {Nu_val:.6f}")
    logger.info("==================================================")

    return save_filename

if __name__ == "__main__":
    solve_steady_shooting("state_Ra10000.npy", delta_T=100)
