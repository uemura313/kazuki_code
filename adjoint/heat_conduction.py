import numpy as np
import dedalus.public as d3
from scipy.sparse.linalg import LinearOperator, gmres
import logging
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# Parameters and Coordinates
# ==========================================
Lx, Lz = 2.0, 1.0
Nx, Nz = 64, 32  # Lower resolution for quick testing
Rayleigh = 10000.0  # Lower Ra for conduction state
Prandtl = 1.0
dealias = 3/2
dtype = np.float64

coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# ==========================================
# Forward Fields (Thermal Conduction State)
# ==========================================
fwd_fields = [
    dist.Field(name='p', bases=(xbasis,zbasis)),
    dist.Field(name='b', bases=(xbasis,zbasis)),
    dist.VectorField(coords, name='u', bases=(xbasis,zbasis)),
    dist.Field(name='tau_p'),
    dist.Field(name='tau_b1', bases=xbasis),
    dist.Field(name='tau_b2', bases=xbasis),
    dist.VectorField(coords, name='tau_u1', bases=xbasis),
    dist.VectorField(coords, name='tau_u2', bases=xbasis)
]
p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2 = fwd_fields

# Directly assign analytical solution for conduction state
z_grid = dist.local_grid(zbasis)
u['g'] = 0.0          # Zero velocity
b['g'] = Lz - z_grid  # Linear temperature profile
p['g'] = 0.0          # Zero pressure

logger.info("  Set forward state to Analytical Thermal Conduction State.")

def pack_state(fields):
    return np.concatenate([f['c'].flatten() for f in fields])

def unpack_state(x_array, fields):
    offset = 0
    for f in fields:
        size = f['c'].size
        f['c'][:] = x_array[offset : offset+size].reshape(f['c'].shape)
        offset += size

# ==========================================
# Adjoint Fields and Equations
# ==========================================
adj_fields = [
    dist.Field(name='p_star', bases=(xbasis,zbasis)),
    dist.Field(name='b_star', bases=(xbasis,zbasis)),
    dist.VectorField(coords, name='u_star', bases=(xbasis,zbasis)),
    dist.Field(name='tau_p_star'),
    dist.Field(name='tau_b1_star', bases=xbasis),
    dist.Field(name='tau_b2_star', bases=xbasis),
    dist.VectorField(coords, name='tau_u1_star', bases=xbasis),
    dist.VectorField(coords, name='tau_u2_star', bases=xbasis)
]
p_star, b_star, u_star, tau_p_star, tau_b1_star, tau_b2_star, tau_u1_star, tau_u2_star = adj_fields

kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)

grad_u_star = d3.grad(u_star) + ez*lift(tau_u1_star)
grad_b_star = d3.grad(b_star) + ez*lift(tau_b1_star)

# Adjoint equations (u@grad terms evaluate to zero since u=0)
problem = d3.IVP(adj_fields, namespace={**globals(), **locals()})
problem.add_equation("trace(grad_u_star) + tau_p_star = 0")
problem.add_equation("dt(b_star) - kappa*div(grad_b_star) + lift(tau_b2_star) = - u@grad(b_star) - (u@ez) + (u_star@ez)")
problem.add_equation("dt(u_star) - nu*div(grad_u_star) + grad(p_star) + lift(tau_u2_star) = - u@grad(u_star) - grad(u)@u_star - b*ez - b_star*grad(b)")
problem.add_equation("b_star(z=0) = 0")
problem.add_equation("u_star(z=0) = 0")
problem.add_equation("b_star(z=Lz) = 0")
problem.add_equation("u_star(z=Lz) = 0")
problem.add_equation("integ(p_star) = 0")

# ==========================================
# Steady State Search (Shooting Method)
# ==========================================
def run_adjoint_for_deltaT(x_array, delta_T):
    unpack_state(x_array, adj_fields)
    solver = problem.build_solver(d3.RK222)
    solver.sim_time = 0.0
    dt = 0.025

    while solver.sim_time < delta_T - 1e-8:
        step_dt = min(dt, delta_T - solver.sim_time)
        solver.step(step_dt)

    return pack_state(adj_fields)

def compute_shooting_residual(x_array, delta_T):
    x_final = run_adjoint_for_deltaT(x_array, delta_T)
    return x_final - x_array    

def apply_J_shooting(v_array, x_current, current_F, delta_T):
    norm_v = np.linalg.norm(v_array)
    if norm_v < 1e-14:
        return np.zeros_like(v_array)

    # Fix epsilon to a constant to prevent noise amplification
    epsilon = 1e-5
    F_plus = compute_shooting_residual(x_current + epsilon * v_array, delta_T)
    return (F_plus - current_F) / epsilon

class GMRESCallback:
    def __init__(self):
        self.n = 0
    def __call__(self, pr_norm):
        self.n += 1
        logger.info(f"      [GMRES Inner] Iteration {self.n}: Residual = {pr_norm:.4e}")

# ==========================================
# Main Newton-Krylov Solver
# ==========================================
def solve_adjoint_conduction(delta_T=1.0):
    logger.info("==================================================")
    logger.info(f" Starting Adjoint Time-Stepper JFNK (Conduction)")
    logger.info(f" Using delta_T = {delta_T}")
    logger.info("==================================================")

    N_size = pack_state(adj_fields).size
    x_k = np.zeros(N_size)  # Initial guess (zeros)

    for i in range(15):  # Outer Newton iteration
        current_F = compute_shooting_residual(x_k, delta_T)
        b_array = -current_F

        residual_norm = np.linalg.norm(b_array) / np.sqrt(N_size)
        logger.info(f"  Newton Iteration {i}: RMS Residual = {residual_norm:.4e}")

        if residual_norm < 1e-12:
            logger.info("    -> Adjoint Steady State Converged successfully!")
            break

        matvec = lambda v: apply_J_shooting(v, x_k, current_F, delta_T)
        J_op = LinearOperator((N_size, N_size), matvec=matvec)

        logger.info("    Solving inner GMRES...")
        # Intentional early truncation: loose rtol, restrict maxiter/restart
        delta_x, exit_code = gmres(J_op, b_array, rtol=1e-1, restart=5, maxiter=1, callback=GMRESCallback())

        x_k = x_k + delta_x
        gc.collect()

if __name__ == "__main__":
    solve_adjoint_conduction(delta_T=1.0)
