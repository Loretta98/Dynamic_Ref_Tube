# Approximation of a PDE system with the method of lines 
# Accuracy on number of spatial elements and time domain discretization is tested 
# A good parameter for evaluation is the Courant-Friedrichs-Lewy condition 

from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import casadi 

def solve_pde(N, dt, T_end):
    # Discretization of space 
    x = np.linspace(0, 1, N)
    dx = x[1] - x[0]
    u0 = np.sin(np.pi * x)
    # Check convergence by the CFL criteria
    CFL = (dt*2*(np.pi**2))/(dx)
    if CFL > 1.5: 
        print("N, CFL, dt" ,N,CFL,dt )

    # Symbolic variables for CasaDi
    t = SX.sym('t')  # Time
    u = SX.sym('u', N)  # Spatial variable

    # PDE -> ODEs using central finite difference method 
    dudt = SX.zeros(N)
    for i in range(1, N - 1):
        dudt[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx**2 # discretization of the second order differential equation 

    # Boundary conditions
    dudt[0] = 0  # Dirichlet BC at x = 0
    dudt[-1] = (u[-1] - u[-2]) / dx + np.pi * exp(-t)  # Neumann BC at x = 1

    # CasADi integrator
    pi_squared = np.pi**2
    # Definition od the differential equation to integrate in time 
    ode = {'x': u, 't': t, 'ode': dudt / pi_squared}
    integrator = casadi.integrator('integrator', 'cvodes', ode, {'tf': dt})

    # Time integration
    t_values = np.arange(0, T_end, dt)
    u_results = [u0]
    current_u = u0

    for t in t_values[1:]:
        result = integrator(x0=current_u)
        current_u = result['xf'].full().flatten()
        # Boundary condition are to be applied for each integration 
        current_u[0] = 0  # Apply Dirichlet BC at x = 0
        current_u[-1] = current_u[-2] - dx * np.pi * np.exp(-t)  # Apply Neumann BC at x = 1
        u_results.append(current_u)

    return x, t_values, np.array(u_results)

# Batch processing for arrays of N and dt
def batch_solve(N_array, dt, T_end):
    results = []
    for N in N_array:
        x, t_values, u_results = solve_pde(N, dt, T_end)
        results.append((N, x, t_values, u_results))
    return results

# Example parameters
N_array = [10, 20, 40, 60, 80, 100]  # Array of N values
# Courant Friedrichs - Lewy (CFL) condition is satisfied dt<=dx2/2pi2
dt = 0.1
T_end = 2.0

# Solve for multiple N values
batch_results = batch_solve(N_array, dt, T_end)

# Plot results
plt.figure(figsize=(10, 6))
for N, x, t_values, u_results in batch_results:
    plt.plot(x, u_results[-1], label=f'N={N}')
    
plt.plot(x, np.exp(-2.0) * np.sin(np.pi * x), 'k--', label='Analytical')
plt.xlabel('x')
plt.ylabel('u(x, t_end)')
plt.title(f'Solution Comparison for dt={dt}')
plt.legend()
plt.grid()
plt.show()
