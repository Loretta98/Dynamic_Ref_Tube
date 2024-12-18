from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from Collocation import coll_setup

# -----------------------------------------------------------------------------
# Collocation setup
# -----------------------------------------------------------------------------
nicp = 1        # Number of collocation points per control interval
nk = 1          # Number of elements
tf = 2          # End time
L = 1           # Space Domain
deg = 9         # Degree of polynomial (higher for accuracy)
cp = "radau"

# Collocation setup function (from your Collocation module)
B, C, D, tau_root, h, tau, S = coll_setup(nicp, nk, tf, deg, cp)

# -----------------------------------------------------------------------------
# Model setup
# -----------------------------------------------------------------------------
# Parameters and variables
t = SX.sym("t")                           # Time
x = SX.sym("x", deg + 1)                  # State variable at collocation points
xt = SX.zeros(deg + 1)                    # Time derivative of state variable
xdd = SX.zeros(deg + 1)                   # Second spatial derivative of state variable
xd = SX.zeros(deg+1)
# PDE constants
pi2 = np.pi**2

# ODE system assembly
for j in range(deg + 1):  # Loop over collocation points
    xd[j] = sum(C[k][j] * x[k] for k in range(deg + 1))  # first derivative
    xdd[j] = sum(B[k][j] * x[k] for k in range(deg + 1))  # Second spatial derivative

# Boundary conditions
ode = xt - xdd / pi2  # PDE at internal points
ode[0] = x[0]  # Dirichlet BC: u(0, t) = 0
ode[-1] = sum(C[k][-1] * x[k] for k in range(deg + 1)) + np.pi * exp(-t)  # Neumann BC

# -----------------------------------------------------------------------------
# Solve the system using integrator
# -----------------------------------------------------------------------------
dae = {
    'x': x,
    't': t,
    'ode': ode,
}

integrator = integrator('integrator', 'idas', dae, {'tf': 0.0001})

# Initial condition: u(x, 0) = sin(pi * x)
space_domain = np.linspace(0, L, deg + 1)
u0 = np.sin(np.pi * space_domain)
time_points = np.linspace(0, tf, 1000)

# Time-stepping solution
u_results = []
current_u = u0

for t_i in time_points:
    result = integrator(x0=current_u)
    current_u = result['xf'].full().flatten()
    current_u[0] = 0  # Dirichlet BC
    current_u[-1] = current_u[-2] - h * np.pi * np.exp(-t_i)  # Neumann BC
    u_results.append(current_u)

# -----------------------------------------------------------------------------
# Plot the solution
# -----------------------------------------------------------------------------
X, T = np.meshgrid(space_domain, time_points)
u_results = np.array(u_results)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, T, u_results, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('Time (t)')
ax.set_zlabel('u(x, t)')
ax.set_title('Solution of the PDE over time (Orthogonal Collocation)')
fig.colorbar(surface, label='u(x, t)')
plt.show()
