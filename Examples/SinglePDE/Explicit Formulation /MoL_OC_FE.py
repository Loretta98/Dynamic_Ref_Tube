# Resolution of the following single PDE: pi**2 du/dt = d2u/dx2 
# Example from https://uk.mathworks.com/help/matlab/math/solve-single-pde.html
# Domain 0<=x<=1, 
# Boundary conditions: u(x,0) = sin(pi*x), u(0,t) = 0, pi*e-t+du/dx(1,t) = 0
# Single Element

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
nk = 2          # Number of elements
tf = 2          # End time 
L = 1           # Space Domain
ndstate = 1     # State variables u
nastate = 0     # Algebraic state variable zC 
deg = 3 
cp = "radau"

B,C,D,tau_root,h,tau,S = coll_setup(nicp,nk,tf,deg,cp)

# -----------------------------------------------------------------------------
# Model setup
# -----------------------------------------------------------------------------
# Constant parameters 
# State variables 
t = SX.sym("t")             # Time
x = SX.sym("x",(deg+1,nk-1))           # Differential state (u) time discretization = n --> n differential states 
xa = SX.sym("xa",(nk-2))         # ALgebraic state --> continuity equations between each element
xt = SX.zeros((deg+1,nk-1))         # Differential time derivative 
xd = SX.zeros((deg+1,nk-1))         # Differential state space derivative 
xdd = SX.zeros((deg+1,nk-1))       # Differentail state space second derivative 

p = SX.sym("p",0)           # Symbolic parameter
u = SX.sym("u",0)           # Control Actions 

# du0/dt = d2u0/dx2     x [0,h0]
# du1/dt = d2u1/dx2     x [h0,h1]

# Reshape x into a 2D structure for calculations
x_matrix = reshape(x, deg+1, nk-1)

for k in range(0, nk-1):
    for i in range(nicp):
        for j in range(0, deg+1):
            for j2 in range(deg+1):
                xd[j + k * (deg+1)] += C[j2][j] * x_matrix[j2, k]
                xdd[j + k * (deg+1)] += B[j2][j] * x_matrix[j2, k]
            if 0 < k < nk-2:
                xa[k-1] = D[j] * x_matrix[0, k]

xt[-1] = xd[-1] - np.pi*np.exp(-t)
xt[0] = 0 
alg = xa
ode  = xt - xdd/np.pi**2

# The integrator is here recalled for integration of the DAE system 
space_domain = np.linspace(0, L, (deg+1) * (nk-1))
u0 = np.ones(np.size(space_domain))*np.sin(np.pi*space_domain)
time_points = np.linspace(0, tf, 100)  # Intermediate time points

# Correct Implementation of the BC 
u_results = []  # Store results for all time steps
current_u = u0 # Initial condition

dae = {
    'x': x,  # Pass flattened x
    't':t,
    'z': vertcat(xa),  # Algebraic state remains a vector
    'p': vertcat(p),   # Parameters
    'ode': reshape(xt, (deg+1)*(nk-1),1),  # Flatten ode back to a vector
    'alg': alg
}

integrator = integrator('integrator','idas',dae, {'tf':0.01})

for t in time_points:
    result = integrator(x0=current_u)
    current_u = result['xf'].full().reshape(-1)  # Flattened result
    u_matrix = current_u.reshape((deg+1, nk-1))  # Reshape with explicit dimensions
    u_results.append(u_matrix.flatten())  # Append in flattened form

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

# Loop through spatial points and plot evolution over time
for i in range():
    ax.plot(time_points[1:], [x[i]] * len(time_points[1:]), u_results[:, i], label=f'u_{i}', alpha=0.8)

ax.set_xlabel('Time (t)')
ax.set_ylabel('x')
ax.set_zlabel('u(x, t)')
ax.set_title('Evolution of u_i(t) for Each Spatial Point')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Spatial Points")
plt.show()

# # For all finite elements
# for k in range(0,nk-1):
#     for i in range(nicp):
#         # For all collocation points
#         for j in range(0,deg+1): 	
#             # Get an expression for the state derivative at each collocation point 
#             for j2 in range (deg+1): 
#                 xd[k] += C[j2][j]*x[j,k]        # get the time derivative of the differential states (eq 10.19b)
#                 xdd[k] += B[j2][j]*x[j,k]        # get the second time derivate of the differential states 
#             if 0 < k < nk-2: 
#                 # Continuity equation between each element 
#                 xa[k-1] = D[j]*x[0,k+1]  

#     xt[k] = vertcat(xdd[k]/np.pi**2)