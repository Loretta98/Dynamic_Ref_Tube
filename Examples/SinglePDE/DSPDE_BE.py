from casadi import *
import numpy as np
import matplotlib.pyplot as plt

# Discretization
N = 10  # Number of spatial points
x = np.linspace(0, 1, N)
dx = x[1] - x[0]  # Spatial step size

# Time stepping
T_end = 2.0  # Final time
num_time_steps = 100
dt = T_end / num_time_steps  # Time step size
time_points = np.linspace(0, T_end, num_time_steps + 1)

# Initial condition: u(x, 0) = sin(pi * x)
u0 = np.sin(np.pi * x)

# PDE parameters
pi_sq = np.pi**2

# Symbolic variables for solving implicit system
u_n1 = SX.sym('u_n1', N)  # Solution at time n+1
u_n = SX.sym('u_n', N)    # Solution at time n (previous step)
t_sym = SX.sym('t')       # Time variable for boundary condition

# Residual array
residual = SX.zeros(N)

# Interior points (finite difference for second derivative)
for i in range(1, N - 1):
    residual[i] = (u_n1[i] - u_n[i]) - dt * (
        (u_n1[i+1] - 2 * u_n1[i] + u_n1[i-1]) / dx**2
    )

# Boundary conditions
# Dirichlet BC at x = 0: u(0, t) = 0
residual[0] = u_n1[0]

# Neumann BC at x = 1: du/dx = -pi * exp(-t)
residual[-1] = (
    (u_n1[-1] - u_n1[-2]) / dx
    + np.pi * exp(-t_sym)
)

# Define the residual as a CasADi function
residual_function = Function('residual', [u_n1, u_n, t_sym], [residual])

# Create the solver for the implicit equation
solver = rootfinder('solver', 'newton', {'x': u_n1, 'p': vertcat(u_n, t_sym), 'g': residual})

# Time stepping loop
u_current = u0  # Initial condition
solution = [u0]  # Store the solution for each time step

for t in time_points[1:]:
    # Solve for u_next
    u_next = solver(p=DM(np.hstack((u_current, t))))  # Ensure compatibility with CasADi
    u_current = np.array(u_next).flatten()  # Update current solution
    solution.append(u_current)
    
# Convert solution to array for plotting
solution = np.array(solution)

# Plot the final solution and analytical solution at T_end
plt.figure()
plt.plot(x, solution[-1], 'o-', label='Numerical Solution')
plt.plot(x, np.exp(-time_points[-1]) * np.sin(np.pi * x), 'r--', label='Analytical Solution')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('Backward Euler Solution at T_end')
plt.legend()
plt.show()

# Plot the solution over time
X, T = np.meshgrid(x, time_points)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, T, solution, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('Time (t)')
ax.set_zlabel('u(x, t)')
ax.set_title('Solution of the PDE over time (Backward Euler)')
plt.show()
