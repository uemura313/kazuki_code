# ==============================================================================
# Physical Parameters
# ==============================================================================
Ra = 10**3.7         # Rayleigh number (Ra)
Pr = 1.0             # Prandtl number (Pr)

# ==============================================================================
# Domain & Grid Parameters
# ==============================================================================
Nx = 64               # Horizontal resolution
Nz = 32               # Vertical resolution
Lx = 2.0              # Horizontal domain length (width)
Lz = 1.0              # Vertical domain height (depth)

# ==============================================================================
# Time-Stepping & Solver Parameters
# ==============================================================================
dt = 0.025           # DNS time step
T_max = 2000 * dt    # Flow map interval (T) for Newton-Krylov
F_TOL = 1e-5         # Residual tolerance for GMRES convergence
MAX_ITER = 20        # Maximum Newton iterations
INITIAL_GUESS_FILE = "initial_guess.npy"
#INITIAL_GUESS_FILE = "steady_state_Ra_2.00e+03.npy"
