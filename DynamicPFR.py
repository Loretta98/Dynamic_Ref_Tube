# The code starts from solving a symple PDE system with both spatial and time as integration variables 
# First goal is to find the steady state solution or very close to it to test the integration

# It is implemented the method of lined rather than the finite difference one on both time and space. 

from casadi import *
import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 9.8e-5  # Pressure drop coefficient [1/kg]
FAin = 50       # Inlet molar flow rate of A [mol/min]
Pin = 40        # Pressure [atm]
xAin = 0.30     # Inlet molar fraction of A
k = 0.00087     # Kinetic parameter [mol/atm^2/kgcat/min]
Ka = 1.39       # Kinetic parameter [1/atm]
Kc = 1.038      # Kinetic parameter [1/atm]
PAin = Pin * xAin
Wmax = 10       # Catalyst weight [kg]
u = 1           # Superficial velocity [kg/min]

# Discretization
N = 100          # Number of spatial grid points
W = np.linspace(0, Wmax, N)
dW = W[1] - W[0]

# Initial conditions
X0 = np.zeros(N)  # Initial conversion at all grid points
X0[0] = 0         # Boundary condition at W=0 (inlet)

# Symbolic variables
t = SX.sym('t')            # Time
X = SX.sym('X', N)         # Conversion along spatial domain

# Reaction rate
def reaction_rate(X, k, Kc, Ka, PAin):
    PA = PAin * (1 - X)
    PB = PAin * (1.5 - X)
    PC = PAin * X
    r = k * PA * PB / (1 + Kc * PC + Ka * PA)
    return r

# PDE -> ODEs using finite differences
r = reaction_rate(X, k, Kc, Ka, PAin)
dX_dt = SX.zeros(N)
for i in range(1, N):
    dX_dt[i] = -u * (X[i] - X[i-1]) / dW + r[i] / FAin
dX_dt[0] = 0  # Boundary condition at inlet

# CasADi integrator
ode = {'x': X, 't': t, 'ode': dX_dt}
integrator = integrator('integrator', 'cvodes', ode, {'tf': 1.0})  # Set default tf

# Time integration
T_end = 2     # Simulation time
dt = 0.01       # Time step
t_values = np.arange(0, T_end, dt)
X_results = [X0]
current_X = X0

for t in t_values[1:]:
    result = integrator(x0=current_X)  # Solve for next time step
    current_X = result['xf']
    X_results.append(current_X.full().flatten())

# Convert results to numpy array
X_results = np.array(X_results)

# Plot evolution over time
plt.figure(figsize=(8, 6))
plt.contourf(W, t_values, X_results, levels=50, cmap='viridis')
plt.colorbar(label='Conversion (X)')
plt.xlabel("Weight of Catalyst (W) [kg]")
plt.ylabel("Time (t) [min]")
plt.title("Evolution of Conversion Over Time in PFR")
plt.grid()
plt.show()


