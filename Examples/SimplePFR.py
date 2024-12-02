# Simple space integration in casadi for a Plug Flow Reactor
# Example from acuoci packed bed reactor in github

from casadi import *
import numpy as np
import matplotlib.pyplot as plt

# Define ODE system
def ode(w, x, k, Kc, Ka, PAin, FAin):
    # Partial pressures of species
    PA = PAin * (1 - x)
    PB = PAin * (1.5 - x)
    PC = PAin * x
    
    # Overall reaction rate
    rprime = k * PA * PB / (1 + Kc * PC + Ka * PA)
    
    # ODE: dX/dW
    dX_over_dW = rprime/FAin
    return dX_over_dW

# Parameters from MATLAB code
alpha = 9.8e-5  # pressure drop coefficient [1/kg]
FAin = 50       # inlet molar flow rate of A [mol/min]
Pin = 40        # inlet pressure [atm]
T = 640         # temperature [C] (not used in this model)
rhoB = 400      # bed density [kg/m3] (not used in this model)
xAin = 0.30     # inlet molar fraction of A
xBin = 0.45     # inlet molar fraction of B
xIin = 0.25     # inlet molar fraction of inerts
k = 0.00087     # kinetic parameter [mol/atm^2/kgcat/min]
Ka = 1.39       # kinetic parameter [1/atm]
Kc = 1.038      # kinetic parameter [1/atm]

# Maximum catalyst weight
Pmin = 1                         # minimum pressure [atm]
Wmax = (1 - (Pmin / Pin) ** 2) / alpha  # maximum amount of catalyst [kg]

# Initial conditions
PAin = Pin * xAin  # inlet partial pressure of A [atm]
X0 = 0.            # initial conversion

# Symbolic variables
w = SX.sym('w')  # Independent variable (space)
x = SX.sym('x')  # Conversion

# ODE definition
dx_dw = ode(w, x, k, Kc, Ka, PAin, FAin)
tf = SX.sym('tf')  # Define final time as a parameter

# CasADi integrator setup
ode = {'x': x, 't': w, 'ode': dx_dw, 'p': tf}
integrator = integrator('integrator', 'cvodes', ode)

# Integration settings
n_points = 1000 # the parameter influences strongly the results 
w_values = np.linspace(0, 0.98 * Wmax, n_points)  # Use 98% of Wmax like MATLAB
x_values = []

# Solve the ODE iteratively
current_x = X0
for i in range(len(w_values) - 1):
    result = integrator(x0=current_x, p=w_values[i + 1] - w_values[i])
    current_x = result['xf']
    x_values.append(float(current_x))

# Append the final point
x_values.append(float(current_x))

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(w_values, x_values, label="Conversion (x)")
plt.xlabel("Weight of Catalyst (W) [kg]")
plt.ylabel("Conversion (X)")
plt.title("Conversion Profile along the Reactor")
plt.grid()
plt.legend()
plt.show()


