from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from acados_template import AcadosModel
from utils import create_casados_integrator, create_casadi_integrator

from casadi import SX, MX
from acados_template import AcadosModel

def validate_acados_model(model):
    # Validate `f_impl_expr` (implicit function)
    if not isinstance(model.f_impl_expr, dict):
        print("Error: `f_impl_expr` is not a dictionary!")
    elif not all(key in model.f_impl_expr for key in ['x', 't', 'ode']):
        print("Error: `f_impl_expr` does not have required keys ['x', 't', 'ode']!")
    else:
        print("`f_impl_expr` is correctly defined.")
    
    # Validate `f_expl_expr` (explicit function)
    if not isinstance(model.f_expl_expr, dict):
        print("Error: `f_expl_expr` is not a dictionary!")
    elif not all(key in model.f_expl_expr for key in ['x', 't', 'f_expl']):
        print("Error: `f_expl_expr` does not have required keys ['x', 't', 'ode']!")
    else:
        print("`f_expl_expr` is correctly defined.")

    # Validate `x` (state variable)
    if not isinstance(model.x, (SX, MX)):
        print("Error: `x` is not of type casadi.SX or casadi.MX!")
    else:
        print("`x` is correctly defined.")

    # Validate `xdot` (state derivative)
    if not isinstance(model.xdot, (SX, MX)):
        print("Error: `xdot` is not of type casadi.SX or casadi.MX!")
    else:
        print("`xdot` is correctly defined.")

    # Validate `u` (control input)
    if not isinstance(model.u, (SX, MX)):
        print("Error: `u` is not of type casadi.SX or casadi.MX!")
    else:
        print("`u` is correctly defined.")

    # Validate `z` (algebraic variables)
    if not (isinstance(model.z, (SX, MX)) or model.z is None):
        print("Error: `z` is not of type casadi.SX, casadi.MX, or None!")
    else:
        print("`z` is correctly defined.")

    # Validate `name` (model name)
    if not isinstance(model.name, str):
        print("Error: `name` is not a string!")
    else:
        print("`name` is correctly defined.")

    # Summary
    print("\nModel validation completed!")

def Model(N): 
    x = np.linspace(0, 1, N)
    dx = x[1] - x[0]  # Spatial step size

    # Symbolic variables
    t = SX.sym('t')          # Time
    u = SX.sym('u', N)       # Variable along spatial domain --> N number of equations 

    # PDE -> ODEs using finite differences
    dudt = SX.sym('dudt',N)  # Initialize time derivative array
    dudx2 = SX.zeros(N)
    dudx = SX.zeros(N)
    # Interior points (finite difference for second derivative) --> implicit expression 
    # Interior points: Finite difference for second derivative
    for i in range(1, N - 1):
        dudx2[i] = (u[i+1] - 2*u[i] + u[i-1]) / dx**2
    # Assemble the system
    pi_squared = np.pi**2
    # Boundary conditions
    dudx2[0] = 0  # Dirichlet at x = 0
    dudx2[-1] = ((u[-1]- u[-2])/dx + np.pi * exp(-t)) # Neumann at x = 1

    ode = vertcat((dudt - dudx2 / pi_squared))#+ dudx))

    # Define explicit dynamics
    f_expl = dudx2 / pi_squared# + dudx 

    z = None
    u_ = None
    model_name = "simplePDE"

    model = AcadosModel()
    model.f_impl_expr = ode
    model.f_expl_expr = f_expl
    model.x = u
    model.xdot = dudt
    model.u = u_
    model.z = z
    model.t = t
    model.name = model_name 
    return model 

# Discretization
N = 13 # Number of spatial points
nt = 15 # Number of temporal points

#validate_acados_model(model)
T_end = 2.0  # Simulation time
time_points = np.linspace(0, T_end, nt)  # Intermediate time points
# Time grid for integration
dt = T_end/nt 
use_cython = False
integrator_opts = {
    "type": "implicit",
    "collocation_scheme": "radau",
    "num_stages": 6,
    "num_steps": 3,
    "newton_iter": 10,
    "tol": 1e-6,
}

# Analytical solution (assumed form for demonstration)
def u_analytical(x, t):
    return np.sin(np.pi * x) * np.exp(-t)

# Error-based grid refinement
errors = []
grid_sizes = range(10, 101, 10 )  # Test different grid sizes
x = np.linspace(0, 1, N)
u0 = np.sin(np.pi * x)
dx = x[1] - x[0]
model = Model(N)
solution_trajectory = np.zeros((N, len(time_points)))  # Preallocate for efficiency
test_integrator = create_casados_integrator(
        model, integrator_opts, dt=dt, use_cython=use_cython)
# for N in grid_sizes:
#     print(N)
#     T_end = 2.0  # Simulation time
#     nt = N 
#     time_points = np.linspace(0, T_end, nt)  # Intermediate time points
#     # Validate the model
#     x = np.linspace(0, 1, N)
#     u0 = np.sin(np.pi * x)
#     dx = x[1] - x[0]
#     model = Model(N)
#     solution_trajectory = np.zeros((N, len(time_points)))  # Preallocate for efficiency
#     test_integrator = create_casados_integrator(
#             model, integrator_opts, dt=dt, use_cython=use_cython
#         )
#     for t0 in time_points:
#         result = test_integrator(x0=u0, p=t0)  # Solve at the current time
#         u0 = result['xf']  # Update initial condition for the next step
#         u0[0] = 0  # Apply Dirichlet BC at x = 0
#         u0[-1] = u0[-2] - dx * np.pi * np.exp(-t0)  # Apply Neumann BC at x = 1
#         solution_trajectory[:, int(t0)] = u0.full().flatten()  # Store the solution

#     # Numerical solution at t=2 (placeholder)
#     u_numerical = solution_trajectory[:, -1]  # Replace with your numerical results
#     u_exact = u_analytical(x, t=2)
    
#     # Compute error
#     error = np.linalg.norm(u_numerical - u_exact, ord=2)
#     errors.append(error)

# # Plot error vs grid size
# plt.plot(grid_sizes, errors, marker='o')
# plt.xlabel('Number of Grid Points (N)')
# plt.ylabel('Error (L2 norm)')
# plt.title('Error vs Grid Size')
# plt.show()

# Multiple shooting
solution_trajectory = []  # Store all time steps
t0 = 0  # Start time
solution_trajectory = np.zeros((N, len(time_points)))  # Preallocate for efficiency

# Multiple shooting
for idx, t0 in enumerate(time_points):
    result = test_integrator(x0=u0, p=t0)  # Solve at the current time
    u0 = result['xf']  # Update initial condition for the next step
    u0[0] = 0  # Apply Dirichlet BC at x = 0
    u0[-1] = u0[-2] - dx * np.pi * np.exp(-t0)  # Apply Neumann BC at x = 1
    solution_trajectory[:, idx] = u0.full().flatten()  # Store the solution
    print(t0,'to')
# Convert solution trajectory to a numpy array for easier plotting
solution_trajectory = np.array(solution_trajectory).T  # Transpose for better visualization

# Plot the solution
X, T = np.meshgrid(x, time_points)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, T, solution_trajectory, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('Time (t)')
ax.set_zlabel('u(x, t)')
ax.set_title('Solution of the PDE over time')

plt.figure()
plt.scatter(x,solution_trajectory[-1])
plt.plot(x,np.exp(-time_points[-1]) * np.sin(np.pi*x))

# Plot each u_i(t) as a line in the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Loop through spatial points and plot evolution over time
for i in range(N):
    ax.plot(time_points, [x[i]] * len(time_points), solution_trajectory[:, i], label=f'u_{i}', alpha=0.8)

ax.set_xlabel('Time (t)')
ax.set_ylabel('x')
ax.set_zlabel('u(x, t)')
ax.set_title('Evolution of u_i(t) for Each Spatial Point')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Spatial Points")
plt.show()

