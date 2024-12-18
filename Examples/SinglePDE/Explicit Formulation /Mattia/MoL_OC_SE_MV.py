# Resolution of the following single PDE: pi**2 du/dt = d2u/dx2 
# Example from https://uk.mathworks.com/help/matlab/math/solve-single-pde.html
# Domain 0<=x<=1, 
# Boundary conditions: u(x,0) = sin(pi*x), u(0,t) = 0, pi*e-t+du/dx(1,t) = 0
# # Method of lines with finite differences  

from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca 
from Collocation import coll_setup
# 1D Orthogonal Spline Collocation Method --> PDE to ODE 

# -----------------------------------------------------------------------------
# Collocation setup
# -----------------------------------------------------------------------------
nicp = 1        # Number of (intermediate) collocation points per control interval
nk = 1          # Number of elements
tf = 2          # End time 
L = 1           # Space Domain
deg = 3        # Since there is only one element, a higher degree is implemented      
cp = "radau"    

B,C,D,tau_root,h,tau,S = coll_setup(nicp,nk,L,deg,cp)

# -----------------------------------------------------------------------------
# Model setup
# -----------------------------------------------------------------------------
# Constant parameters 
# State variables 
t = SX.sym("t")             # Time
x = SX.sym("x",(deg+1))           # Differential state (u) time discretization = n --> n differential states 
xa = SX.sym("xa",0)         # ALgebraic state --> continuity equations between each element

xt = SX.zeros((deg+1))         # Differential time derivative 
xd = SX.zeros((deg+1))         # Differential state space derivative 
xdd = SX.zeros((deg+1))       # Differentail state space second derivative 
rhs = SX.zeros((deg+1))
p = SX.sym("p",0)           # Symbolic parameter
u = SX.sym("u",0)           # Control Actions 

# du0/dt = d2u0/dx2     x [0,h0]
# du1/dt = d2u1/dx2     x [h0,h1]

xt[0] = 0 
# For all collocation points
for j in range(1,deg+1): 	
    # Get an expression for the state derivative at each collocation point 
    xd_v = 0
    xdd_v = 0 
    for i in range (deg+1): 
        xd_v += C[i][j] * x[i]        # get the time derivative of the differential states (eq 10.19b)
        xdd_v += B[i][j] * x[i]        # get the second time derivate of the differential states           
        print(xd)   
    xd[j] = xd_v/h  # duj/dz j = 0, 1,
    xdd[j] = xdd_v/(h**2)
    xt[j] = xdd[j]/np.pi**2

xt[-1] = xd[-1] + np.pi*exp(-t) 
#print('xd',xd), #print('xdd',xdd),
print('xt[-1]',xt[-1])

alg = vertcat() 
#ode  = vertcat(xt-xdd/np.pi**2)
#ode = xt - rhs
ode = vertcat(xt)
# ode[0] = 0                   # Enforce Dirichlet condition at x=0
# ode[-1] = xd[-1] + np.pi*np.exp(-t)  # Enforce Neumann-like condition at x=1

#print(ode[0]), print(ode[4]), print(ode[-1])
# The integrator is here recalled for integration of the DAE system 
space_domain = np.linspace(0, L, (deg+1))
u0 = np.sin(np.pi*space_domain)
#u0 = np.sin(np.pi*x)
n = 1000
time_points = np.linspace(0, tf, n)  # Intermediate time points

# Correct Implementation of the BC 
u_results = []  # Store results for all time steps
current_u = u0 # Initial condition

dae = {
    'x': x,  
    't':t,
    'z': xa,  # Algebraic state remains a vector
    'p': vertcat(p),   # Parameters
    'ode': ode, 
    'alg': alg
}

integrator = integrator('integrator','cvodes',dae, {'tf':tf/n})

for t in time_points:
    result = integrator(x0=current_u)
    current_u = result['xf']  # Flattened results
    current_u[0] = 0  # Dirichlet BC
    current_u[-1] = current_u[-2] - h * np.pi * np.exp(-t) # Apply Neumann BC at x = 1
    u_results.append(current_u)  # Append in flattened form

# Ensure `X`, `T`, and `u_results` are compatible
X, T = np.meshgrid(space_domain, time_points)  # Match time steps
u_results = np.array(u_results).reshape(len(time_points), len(space_domain))
# 3D Surface Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, T, u_results, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('Time (t)')
ax.set_zlabel('u(x, t)')
ax.set_title('Solution of the PDE over time (Orthogonal Collocation)')
fig.colorbar(surface, label='u(x, t)')
plt.show()