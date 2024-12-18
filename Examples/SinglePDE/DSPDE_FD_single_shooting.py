from casadi import *
import numpy as np
import matplotlib.pyplot as plt

# Discretization
N = 10  # Number of spatial points
x = np.linspace(0, 1, N)
dx = x[1] - x[0]  # Spatial step size

# Initial condition: u(x, 0) = sin(pi * x)
u0 = np.sin(np.pi * x)

# Symbolic variables
t = SX.sym('t')          # Time
u = SX.sym('u', N)       # Variable along spatial domain --> N number of equations 

# PDE -> ODEs using finite differences
dudt = SX.zeros(N)  # Initialize time derivative array

# Interior points (finite difference for second derivative) --> implicit expression 
for i in range(1, N - 1):
    dudt[i] = (u[i+1] - 2*u[i] + u[i-1]) / dx**2

# Boundary conditions
dudt[0] = 0  # Dirichlet BC at x = 0: u(0, t) = 0
dudt[-1] = (u[-1] - u[-2]) / dx + np.pi * exp(-t)  # Neumann BC at x = 1: pi*exp(-t) + du/dx(1, t) = 0

# CasADi integrator
pi_squared = np.pi**2
ode = {'x': u, 't': t, 'ode': dudt / pi_squared}

# Time grid for integration
T_end = 2.0  # Simulation time
time_points = np.linspace(0, T_end, 10)  # Intermediate time points

# Define integrator with time grid
integrator = integrator('integrator', 'cvodes', ode, {'grid': time_points.tolist()})

# Solve the system
result = integrator(x0=u0)  # Pass only initial condition

# Retrieve solution
solution_trajectory = result['xf'].full()  # All solutions along the time grid

# Convert to array
solution_trajectory = np.array(solution_trajectory).T  # Transpose for easier use

# Plot the solution
X, T = np.meshgrid(x, time_points)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X[1:], T[1:], solution_trajectory, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('Time (t)')
ax.set_zlabel('u(x, t)')
ax.set_title('Solution of the PDE over time')

plt.figure()
plt.scatter(x,solution_trajectory[-1])
plt.plot(x,np.exp(-time_points[-1])*np.sin(np.pi*x))

# Plot each u_i(t) as a line in the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Loop through spatial points and plot evolution over time
for i in range(N):
    ax.plot(time_points[1:], [x[i]] * len(time_points[1:]), solution_trajectory[:, i], label=f'u_{i}', alpha=0.8)

ax.set_xlabel('Time (t)')
ax.set_ylabel('x')
ax.set_zlabel('u(x, t)')
ax.set_title('Evolution of u_i(t) for Each Spatial Point')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Spatial Points")
plt.show()
