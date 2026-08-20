import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.integrate import solve_ivp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 1. Pseudo-adjoint equation (dx/dt = Ax + b)
# ==========================================
# 'A' acts as a linear operator (like diffusion/advection), 'b' is a steady source term
A = np.array([[-3.0,  1.0,  0.0],
              [ 1.0, -3.0,  1.0],
              [ 0.0,  1.0, -3.0]])
b_vec = np.array([1.0, 2.0, 1.0])

def ode_system(t, x):
    return A @ x + b_vec

# ==========================================
# 2. Time evolution mapping (simulates Dedalus integration)
# ==========================================
def time_evolution_mapping(x_initial, delta_T):
    # Integrate from x_initial for delta_T using RK45
    sol = solve_ivp(ode_system, [0, delta_T], x_initial, method='RK45', rtol=1e-8, atol=1e-8)
    # Return the final state x(delta_T)
    return sol.y[:, -1]

# ==========================================
# 3. Shooting residual function
# ==========================================
def compute_shooting_residual(x_array, delta_T):
    x_final = time_evolution_mapping(x_array, delta_T)
    return x_final - x_array

# ==========================================
# 4. Jacobian-vector product via finite difference
# ==========================================
def apply_J_shooting(v_array, x_current, current_F, delta_T):
    norm_v = np.linalg.norm(v_array)
    if norm_v < 1e-14:
        return np.zeros_like(v_array)

    # Fixed epsilon to prevent noise amplification
    epsilon = 1e-5
    
    # Evaluate R(x + eps*v) (runs one time integration)
    F_plus = compute_shooting_residual(x_current + epsilon * v_array, delta_T)

    # {R(x + eps*v) - R(x)} / eps
    return (F_plus - current_F) / epsilon

# ==========================================
# 5. Main Solver (Shooting JFNK)
# ==========================================
def solve_time_evolution_jfnk(delta_T=1.0):
    logger.info(f"=== Starting Time-Evolution JFNK (delta_T={delta_T}) ===")
    
    x_k = np.zeros(3)  # Initial guess (zeros)
    N_size = len(x_k)

    for i in range(10):  # Outer Newton iteration
        current_F = compute_shooting_residual(x_k, delta_T)
        b_array = -current_F

        residual_norm = np.linalg.norm(b_array)
        logger.info(f"Newton Iteration {i}: RMS Residual = {residual_norm:.4e}")

        if residual_norm < 1e-10:
            logger.info("  -> Newton Converged successfully!")
            break

        matvec = lambda v: apply_J_shooting(v, x_k, current_F, delta_T)
        J_op = LinearOperator((N_size, N_size), matvec=matvec)

        class GMRESCallback:
            def __init__(self):
                self.n = 0
            def __call__(self, pr_norm):
                self.n += 1
                logger.info(f"    [GMRES Inner] Iteration {self.n}: Residual = {pr_norm:.4e}")

        logger.info("  Solving inner GMRES...")
        # Core: Intentionally truncate GMRES early to use an incomplete delta_x
        delta_x, exit_code = gmres(J_op, b_array, rtol=1e-2, restart=2, maxiter=1, callback=GMRESCallback())
        
        x_k = x_k + delta_x

    # Calculate exact steady state (Ax + b = 0) for comparison
    exact_x = np.linalg.solve(A, -b_vec)
    logger.info("==================================================")
    logger.info(f"Final Estimated x = {x_k}")
    logger.info(f"Exact Steady x    = {exact_x}")
    logger.info("==================================================")

if __name__ == "__main__":
    # Changing delta_T alters the mapping length and Jacobian properties
    solve_time_evolution_jfnk(delta_T=0.5)
