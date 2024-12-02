from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre
import casadi as ca 


def legendre_gauss_lobatto(N):
    """Compute Legendre-Gauss-Lobatto points and weights."""
    P = legendre(N - 1)  # Legendre polynomial of degree (N-1)
    x = np.cos(np.linspace(0, np.pi, N))  # Initial guess (Chebyshev points)
    x = np.sort(np.roots(P.deriv()))  # Roots of derivative of Legendre polynomial
    x = np.concatenate([[-1], x, [1]])  # Add -1 and 1 as endpoints
    return x


def differentiation_matrix(x):
    """Compute the differentiation matrix for the collocation points."""
    N = len(x)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (-1) ** (i + j) / (x[i] - x[j])
        D[i, i] = -np.sum(D[i, :])
    return D


# Discretization
N =51   # Number of collocation points
collocation_points = legendre_gauss_lobatto(N)  # Scaled to [-1, 1]
x = 0.5 * (collocation_points + 1)  # Transform to [0, 1]

# Differentiation matrix
D = differentiation_matrix(collocation_points)  # On [-1, 1]
D = 2 * D  # Scale for [0, 1] domain

dx = x[-1]-x[-2]

# Initial condition
u0 = np.sin(np.pi * x)
u0[0] = 0  # Enforce Dirichlet at x=0

# CasADi symbolic variables
t = SX.sym('t')       # Time
u = SX.sym('u', N)    # Variable along collocation points

# Discretized PDE using differentiation matrix
dudt = pi ** 2 * (D @ u)

# Apply boundary conditions directly in the ODE
dudt = vertcat(
    0,  # Dirichlet at x=0: u(0, t) = 0
    dudt[1:-1],  # Internal points
    mtimes(D[-1, :].reshape((1, -1)), u) + np.pi * exp(-t)  # Neumann at x=1
)


# CasADi integrator
ode = {'x': u, 't': t, 'ode': dudt}
T_end = 2.0
dt = 0.0001
t_values = np.arange(0, T_end + dt, dt)

integrator = ca.integrator(
    'integrator',
    'cvodes',
    ode,
    {
        'abstol': 1e-8,         # Set absolute tolerance
        'reltol': 1e-6,          # Set relative tolerance
        'max_num_steps': 1e6,     # Increase max steps for stability
        'max_step_size': 0.01,
        'monitor': ['steps', 'error_test_failures', 'jac_evaluations']
    }
)

# Time integration
current_u = u0
u_results = [current_u]  # Store initial condition

for t1 in t_values[1:]:
    try:
        # Integrate from t0 to t1
        result = integrator(x0=current_u, p=[])
        current_u = result['xf'].full().flatten()  # Extract state
        current_u[0] = 0  # Dirichlet BC at x=0
        current_u[-1] = current_u[-2] - dx * np.pi * np.exp(-t1)  # Neumann BC at x=1
        u_results.append(current_u)
    except RuntimeError as e:
        print(f"Error at t = {t1}: {e}")
        break

u_results = np.array(u_results)  # Convert list of arrays to 2D NumPy array
# Ensure `X`, `T`, and `u_results` are compatible
X, T = np.meshgrid(x, t_values[:len(u_results)])  # Match time steps

# 3D Surface Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, T, u_results, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('Time (t)')
ax.set_zlabel('u(x, t)')
ax.set_title('Solution of the PDE over time (Orthogonal Collocation)')
fig.colorbar(surface, label='u(x, t)')
plt.show()