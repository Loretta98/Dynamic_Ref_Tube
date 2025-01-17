import numpy as np 
from casadi import * 
import matplotlib.pyplot as plt 
# -----------------------------------------------------------------------------
# Model setup
# -----------------------------------------------------------------------------

# Fixed values
k10 =     1.287e12 
k20 =     1.287e12 
k30 =     9.043e9  
H1 =     4.2       
H2 =   -11.0      
H3 =   -41.85      
E1 = -9758.3      
E2 = -9758.3      
E3 = -8560.0      
rho =     0.9342	  
Cp =     3.01	  
kw =  4032.0      
VR =    10.0	
AR = 0.215
mK =	 5.0	  
CPK =	 2.0	  
Ca0 =	 5.1	  
theta0 =   104.9    

# Declare variables (use scalar graph)
t  = SX.sym("t")         # time
u  = SX.sym("u",2)         # control
xd = SX.sym("xd",10)      # differential state
xa = SX.sym("xa",0)      # algebraic state
xddot = SX.sym("xdot",10) # differential state time derivative
p = SX.sym("p",0)        # parameters

k1 = k10*exp(E1/(273.15+xd[2]))
k2 = k20*exp(E2/(273.15+xd[2]))
k3 = k30*exp(E3/(273.15+xd[2]))

# ODE right hand side function
rhs = vertcat( (1.0/3600.0)*(u[0]*(Ca0-xd[0]) - k1*xd[0] - k3*xd[0]*xd[0]),\
        (1.0/3600.0)*(-u[0]*xd[1] + k1*xd[0] - k2*xd[1]),\
        (1.0/3600.0)*(u[0]*(theta0 - xd[2]) - (1.0/(rho*Cp))*(k1*xd[0]*H1 + k2*xd[1]*H2 + k3*xd[0]*xd[0]*H3) + (kw*AR/(rho*Cp*VR))*(xd[3]-xd[2])),\
        (1.0/3600.0)*((1.0/(mK*CPK))*(u[1]+kw*AR*(xd[2]-xd[3]))),\
        pow((xd[0]-2.14),2),\
        pow((xd[1]-1.09),2),\
        pow((xd[2]-114.2),2),\
        pow((xd[3]-112.9),2),\
        pow((u[0]-14.19),2),\
        pow((u[1]-(-1113.5)),2))


# AE right hand side function       
rhsalg = []       
       
# System dynamics (implicit formulation)
ffcn = Function("ffcn",[t,xddot,xd,xa,u,p],[xddot - rhs])

# Algebraic equation
fafcn = Function("fafcn",[t,xd,xa,u,p],[rhsalg])

# Define ODE function
ode = {'x': xd, 'p': u, 'ode': rhs}
opts = {'tf': 10.0}  # Simulation time for a single step
integrator_fn = integrator('integrator_fn', 'cvodes', ode, opts)

# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------

# Initial conditions
x0 = np.zeros(10)  # Replace with appropriate initial values
u_val = [14.19, -1113.5]  # Example control inputs
sim_time = 100  # Total simulation time
dt = 10  # Time step

# Time and state storage
time = []
states = []

# Simulate over time
x_current = x0
for t in np.arange(0, sim_time, dt):
    result = integrator_fn(x0=x_current, p=u_val)
    x_current = result['xf'].full().flatten()  # Update state
    time.append(t)
    states.append(x_current)

# Convert results to arrays
time = np.array(time)
states = np.array(states)

# -----------------------------------------------------------------------------
# Plot results
# -----------------------------------------------------------------------------

plt.figure(figsize=(12, 8))
for i in range(4):  # Plot key states (xd[0] to xd[3])
    plt.plot(time, states[:, i], label=f"xd[{i}]")
plt.xlabel("Time [s]")
plt.ylabel("States")
plt.legend()
plt.grid()
plt.title("Dynamic Reactor Simulation")
plt.show()