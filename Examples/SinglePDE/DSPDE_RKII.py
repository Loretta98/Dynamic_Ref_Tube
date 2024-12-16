from casadi import *
import numpy as np
import matplotlib.pyplot as plt

# Discretization
N = 51  # Number of spatial points
x = np.linspace(0, 1, N)
dx = x[1] - x[0]  # Spatial step size

# Time stepping
T_end = 2.0  # Final time
num_time_steps = 100
dt = T_end / num_time_steps  # Time step size
time_points = np.linspace(0, T_end, num_time_steps + 1)

# Initial condition: u(x, 0) = sin(pi * x)
u0 = np.sin(np.pi * x)

# Radau-IIA Butcher Tableau
A = np.array([[5/12, -1/12],
              [3/4,  1/4]])
b = np.array([3/4, 1/4])
c = np.array([1/3, 1])

# Symbolic variables
u_n = SX.sym('u_n', N)         # Solution at current time step
K = SX.sym('K', 2, N)          # Stage values (2 stages, N spatial points)

# Define residuals for each stage
residuals = SX.zeros((2, N))
for i in range(2):
    for j in range(2):
        residuals[i, :] += A[i, j] * K[j, :]
    residuals[i, :] = K[i, :] - (1 / np.pi**2) * (dt * residuals[i, :])
    
    # Interior finite differences
    for k in range(1, N - 1):
        residuals[i, k] -= (K[i, k+1] - 2*K[i, k] + K[i, k-1]) / dx**2

    # Boundary conditions
    residuals[i, 0] = 0  # Dirichlet BC at x = 0
    residuals[i, -1] -= (K[i, -1] - K[i, -2]) / dx + np.pi * exp(-(c[i] * dt))

# Flatten residuals into a dense column vector
residuals_flat = reshape(residuals, -1, 1)  # Flatten to a single column

# Define solver for IRK system
solver = rootfinder('solver', 'newton', {'x': reshape(K, -1, 1), 'p': u_n, 'g': residuals_flat})

# Time stepping loop
u_current = u0  # Initial condition
solution = [u0]  # Store the solution for each time step

for t in time_points[1:]:
    # Solve for stage values K (reshaped into a flat vector)
    K_flat = solver(p=u_current)
    K_matrix = reshape(K_flat, 2, N)  # Reshape back to 2 stages × N spatial points

    # Update solution u_{n+1}
    u_next = u_current + dt * (b[0] * K_matrix[0, :] + b[1] * K_matrix[1, :])
    u_current = u_next
    solution.append(u_current)

# Convert solution to array for plotting
solution = np.array(solution)

# Plot the solution
X, T = np.meshgrid(x, time_points)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, T, solution, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('Time (t)')
ax.set_zlabel('u(x, t)')
ax.set_title('Solution of the PDE over time (Implicit Runge-Kutta)')
plt.show()
