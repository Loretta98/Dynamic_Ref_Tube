# Resolution of the following single PDE: pi**2 du/dt = d2u/dx2 
# Example from https://uk.mathworks.com/help/matlab/math/solve-single-pde.html
# Domain 0<=x<=1, 
# Boundary conditions: u(x,0) = sin(pi*x), u(0,t) = 0, pi*e-t+du/dx(1,t) = 0

from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Discretization
N = 11  # Number of spatial points
x = np.linspace(0, 1, N)
dx = x[1] - x[0]  # Spatial step size

# Initial condition: u(x, 0) = sin(pi * x)
u0 = np.sin(np.pi * x)

# Symbolic variables
t = SX.sym('t')          # Time
u = SX.sym('u', N)       # Variable along spatial domain

# PDE -> ODEs using finite differences
dudt = SX.zeros(N)  # Initialize time derivative array

# Interior points (finite difference for second derivative)
for i in range(1, N - 1):
    dudt[i] = (u[i+1] - 2*u[i] + u[i-1]) / dx**2

# Boundary conditions
dudt[0] = 0  # Dirichlet BC at x = 0: u(0, t) = 0
dudt[-1] = (u[-1] - u[-2]) / dx + np.pi * exp(-t)  # Neumann BC at x = 1: pi*exp(-t) + du/dx(1, t) = 0

# CasADi integrator
pi_squared = np.pi**2
ode = {'x': u, 't': t, 'ode': dudt / pi_squared}
integrator = integrator('integrator', 'cvodes', ode, {'tf': 0.01})  # Time step (dt)

# Time integration
T_end = 2.0  # Simulation time
dt = 0.01    # Time step
t_values = np.arange(0, T_end, dt)
u_results = [u0]
current_u = u0

# for t in t_values[1:]:
#     result = integrator(x0=current_u)
#     current_u = result['xf']
#     u_results.append(current_u.full().flatten())

# Correct Implementation of the BC 
for t in t_values[1:]:
    result = integrator(x0=current_u)
    current_u = result['xf']
    current_u[0] = 0  # Apply Dirichlet BC at x = 0
    current_u[-1] = current_u[-2] - dx * np.pi * np.exp(-t)  # Apply Neumann BC at x = 1
    u_results.append(current_u.full().flatten())

# Convert results to numpy array
u_results = np.array(u_results)

# Plot evolution with a surface plot
X, T = np.meshgrid(x, t_values)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, T, u_results, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('Time (t)')
ax.set_zlabel('u(x, t)')
ax.set_title('Solution of the PDE over time')
fig.colorbar(surface, label='u(x, t)')

plt.figure()
plt.scatter(x,u_results[-1])
plt.plot(x,np.exp(-t_values[-1])*np.sin(np.pi*x))
plt.show()


