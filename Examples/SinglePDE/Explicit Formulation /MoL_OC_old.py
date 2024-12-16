# Resolution of the following single PDE: pi**2 du/dt = d2u/dx2 
# Example from https://uk.mathworks.com/help/matlab/math/solve-single-pde.html
# Domain 0<=x<=1, 
# Boundary conditions: u(x,0) = sin(pi*x), u(0,t) = 0, pi*e-t+du/dx(1,t) = 0
# Full orthogonal discretization method 

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
nk = 4          # Number of elements
tf = 2          # End time 
L = 1           # Space Domain
ndstate = 1     # State variables u
nastate = 0     # Algebraic state variable zC 
deg = 3 
cp = "radau"

B,C,D,tau_root,h,tau,S = coll_setup(nicp,nk,tf,ndstate,nastate,deg,cp)

# -----------------------------------------------------------------------------
# Model setup
# -----------------------------------------------------------------------------
# Constant parameters 
# State variables 
t = SX.sym("t")             # Time
z = SX.sym("z")             # Space 
n = nk*deg                  # Total Number of collocation points along the space domain --> but if we consider that the space is discretized over nk elements, within each element the u can be explicitally expressed  
x = SX.sym("x",nk-1)           # Differential state (u) time discretization = n --> n differential states 
xa = SX.sym("xa",nk)         # ALgebraic state --> continuity equations between each element
xt = SX.sym("xt",nk-1)         # Differential time derivative 
xd = SX.sym("xd",nk-1)         # Differential state space derivative 
xdd = SX.sym("xdd",nk-1)       # Differentail state space second derivative 
p = SX.sym("p",0)           # Symbolic parameter
u = SX.sym("u",0)           # Control Actions 


# xd = SX.zeros(nk-1)  # Ensure symbolic initialization
# xdd = SX.zeros(nk-1)
# xa = SX.zeros(nk-1)

# du0/dt = d2u0/dx2     x [0,h0]
# du1/dt = d2u1/dx2     x [h0,h1]
# For all finite elements
for k in range(1,nk-1):
    if k < nk : 
        for i in range(nicp):
            # For all collocation points
            for j in range(1,deg+1): 	
                # Get an expression for the state derivative at each collocation point 
                for j2 in range (deg+1): 
                    xd[k] += C[j2][j]*x[k]        # get the time derivative of the differential states (eq 10.19b)
                    xdd[k] += B[j2][j]*x[k]        # get the second time derivate of the differential states
                    if k == 0: 
                        xa[k] = 0 
                    elif k == (nk-1): 
                        xa[k] = xd[k] - np.pi* ca.exp(-t) # Non sono certa che me la azzeri però 
                        print(type(xa[k]), type(xd[k]), type(ca.exp(-t)))                
                    else: 
                        print(k)
                        xa[k] = D[j]*x[k]

alg = xa
pi_squared = np.pi**2
rhs = vertcat(xdd/pi_squared)
ode  = xt - rhs

# The integrator is here recalled for integration of the DAE system 
space_domain = np.array(S)
u0 = np.ones(np.size(nk))*np.sin(np.pi*space_domain)
time_points = np.linspace(0, tf, 100)  # Intermediate time points

# Correct Implementation of the BC 
u_results = []  # Store results for all time steps
current_u = u0 # Initial condition

for dt in time_points[1:]:
    # -- Recall the NLP for the orthogonal collocation --- 
    u_init = current_u
    dae = {
    'x': vertcat(x),  # Differential states
    'z': vertcat(xa),   # Algebraic state
    'p': vertcat(p),  # Parameters (symbolic)
    'ode': ode,
    'alg': alg
    }
    integrator = integrator('integrator','idas',dae, {'tf': dt.tolist()})
    result = integrator(x0=current_u)
    current_u = result['xf']
    u_results.append(current_u.full().flatten()) 


t_values = np.linspace(0,2)
# Ensure `X`, `T`, and `u_results` are compatible
X, T = np.meshgrid(x, t_values[:len(u_results)])  # Match time steps
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