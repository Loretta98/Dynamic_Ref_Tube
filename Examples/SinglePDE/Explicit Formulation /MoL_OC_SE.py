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
deg = 9        # Since there is only one element, a higher degree is implemented      
cp = "legendre"    

B,C,D,tau_root,h,tau,S = coll_setup(nicp,nk,tf,deg,cp)

# -----------------------------------------------------------------------------
# Model setup
# -----------------------------------------------------------------------------
# Constant parameters 
# State variables 
t = SX.sym("t")             # Time
x = SX.sym("x",(deg+1))           # Differential state (u) time discretization = n --> n differential states 
xa = SX.sym("xa",0)         # ALgebraic state --> continuity equations between each element

# xt = SX.sym("xt",(deg+1))         # Differential time derivative 
# xd = SX.sym("xd",(deg+1))         # Differential state space derivative 
# xdd = SX.sym("xdd",(deg+1))       # Differentail state space second derivative 

xt = SX.zeros((deg+1))         # Differential time derivative 
xd = SX.zeros((deg+1))         # Differential state space derivative 
xdd = SX.zeros((deg+1))       # Differentail state space second derivative 

p = SX.sym("p",0)           # Symbolic parameter
u = SX.sym("u",0)           # Control Actions 

# du0/dt = d2u0/dx2     x [0,h0]
# du1/dt = d2u1/dx2     x [h0,h1]

for i in range(nicp):
    # For all collocation points
    for j in range(1,deg+1): 	
        # Get an expression for the state derivative at each collocation point 
        for j2 in range (deg+1): 
            xd += C[j2][j] * x        # get the time derivative of the differential states (eq 10.19b)
            xdd += B[j2][j] * x        # get the second time derivate of the differential states              

xt[-1] = xd[-1] - np.pi*np.exp(-t)
xt[0] = x[0] 
#alg = vertcat(xt[-1], np.pi*np.exp(-t)+xd[-1])
alg = vertcat() 
ode  = vertcat(xt-xdd/np.pi**2)
#ode[0] = 0                   # Enforce Dirichlet condition at x=0
#ode[-1] = xd[-1] + np.pi*np.exp(-t)  # Enforce Neumann-like condition at x=1

# The integrator is here recalled for integration of the DAE system 
space_domain = np.linspace(0, L, (deg+1))
u0 = np.ones(np.size(space_domain))*np.sin(np.pi*space_domain)
time_points = np.linspace(0, tf, 100)  # Intermediate time points

# Correct Implementation of the BC 
u_results = []  # Store results for all time steps
current_u = u0 # Initial condition

# jac = ca.jacobian(vertcat(ode, alg), vertcat(x, t))
# rank = ca.sparsity(jac).rank()
# print(f"Jacobian structural rank: {rank}")

dae = {
    'x': x,  # Pass flattened x
    't':t,
    'z': xa,  # Algebraic state remains a vector
    'p': [],   # Parameters
    'ode': ode,  # Flatten ode back to a vector
    'alg': alg
}

integrator = integrator('integrator','cvodes',dae, {'tf':0.01})

for t in time_points:
    result = integrator(x0=current_u)
    current_u = result['xf'].full().flatten()  # Flattened results
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