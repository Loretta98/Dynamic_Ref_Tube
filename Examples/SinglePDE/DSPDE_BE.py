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

# Symbolic variables
u_n1 = SX.sym('u_n1', N)  # Solution at time n+1
u_n = SX.sym('u_n', N)    # Solution at time n (previous step)

# PDE -> ODEs using finite differences
dudt = SX.zeros(N)  # Initialize time derivative array

# Interior points (finite difference for second derivative)
for i in range(1, N - 1):
    dudt[i] = (u_n1[i+1] - 2*u_n1[i] + u_n1[i-1]) / dx**2

# Boundary conditions
dudt[0] = 0  # Dirichlet BC at x = 0: u(0, t) = 0
dudt[-1] = (u_n1[-1] - u_n1[-2]) / dx + np.pi * exp(-dt * num_time_steps)  # Neumann BC at x = 1

# Backward Euler system
residual = (u_n1 - u_n) - dt*dudt/(np.pi**2)

# Define the residual as a CasADi function
residual_function = Function('residual', [u_n1, u_n], [residual])

# Create the solver for the implicit equation
solver = rootfinder('solver', 'newton', {'x': u_n1, 'p': u_n, 'g': residual})

# Time stepping loop
u_current = u0  # Initial condition
solution = [u0]  # Store the solution for each time step

for t in time_points[1:]:
    u_next = solver(p=u_current)
    #u_current = u_next.full().flatten()  # Update current solution
    #u_current = u_next['x'].flatten()  # Extract solution and flatten it
    u_current = np.array(u_next['x']).flatten()  # Convert to NumPy array and flatten

    solution.append(u_current)

# Convert solution to array for plotting
solution = np.array(solution)

plt.figure()
plt.scatter(x,solution[-1])
plt.plot(x,np.exp(-time_points[-1])*np.sin(np.pi*x))

# Plot the solution
X, T = np.meshgrid(x, time_points)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, T, solution, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('Time (t)')
ax.set_zlabel('u(x, t)')
ax.set_title('Solution of the PDE over time (Backward Euler)')
plt.show()
