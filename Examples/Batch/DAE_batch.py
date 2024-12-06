# Example from https://gitlab-extern.ivi.fraunhofer.de/philipps/pyomo/-/blob/5.5.1/examples/dae/simulator_dae_example.py
# Batch reactor example from Biegler book on Nonlinear Programming Chapter 9
#
# Inside the reactor we have the first order reversible reactions
#
# A <=> B <=> C
#
# DAE model for the reactor system:
#
# zA' = -p1*zA + p2*zB, zA(0)=1
# zB' = p1*zA - (p2 + p3)*zB + p4*zC, zB(0)=0
# zA + zB + zC = 1
from casadi import * 
import numpy as np 
import matplotlib.pyplot as plt 

# Setup DAE in CasADi 

# Time
t = SX.sym("t")  

# Parameters (symbolic)
p1 = SX.sym("p1")
p2 = SX.sym("p2")
p3 = SX.sym("p3")
p4 = SX.sym("p4")

# Differential state variables
zA = SX.sym("zA")
zB = SX.sym("zB")

# Algebraic state
zC = SX.sym("zC")

# Define ODEs
dzA = -p1 * zA + p2 * zB
dzB = p1 * zA - (p2 + p3) * zB + p4 * zC
ode = vertcat(dzA, dzB)

# Algebraic constraint
alg = zA + zB + zC - 1

# Define DAE system
dae = {
    'x': vertcat(zA, zB),  # Differential states
    'z': zC,              # Algebraic state
    'p': vertcat(p1, p2, p3, p4),  # Parameters (symbolic)
    'ode': ode,
    'alg': alg
}

# Time grid for output
grid = np.linspace(0, 1, 5)  # Time grid from t0=0 to tf=1
t0 = 0
# Create integrator
I = integrator('I', 'idas', dae, {'t0':t0, 'grid': grid})

# Initial conditions
x0 = [1, 0]  # Initial conditions for zA and zB
z0 = [0]     # Initial condition for zC
p = [4, 2, 40, 20]  # Parameter values

# Solve DAE
result = I(x0=x0, z0=z0, p=p)
# Extract solutions
time = grid

# Extract trajectories from the result
zA_sol = result['xf'][0, :].full().flatten()  # zA trajectory
zB_sol = result['xf'][1, :].full().flatten()  # zB trajectory
zC_sol = result['zf'][0, :].full().flatten()  # zC trajectory

if len(time) != len(zA_sol):
    # Manually prepend initial conditions to the solution if missing
    zA_sol = np.insert(zA_sol, 0, x0[0]) 
    zB_sol = np.insert(zB_sol, 0, x0[1]) 
    zC_sol = np.insert(zC_sol,0,z0)
# Plot results
plt.figure(figsize=(8, 6))
plt.plot(time, zA_sol, label="zA", marker="o")
plt.plot(time, zB_sol, label="zB", marker="s")
plt.plot(time, zC_sol, label="zC", marker="^")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Concentration Evolution in Batch Reactor")
plt.legend()
plt.grid()
plt.show()


