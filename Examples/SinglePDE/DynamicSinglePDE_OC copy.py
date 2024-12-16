# Resolution of the following single PDE: pi**2 du/dt = d2u/dx2 
# Example from https://uk.mathworks.com/help/matlab/math/solve-single-pde.html
# Domain 0<=x<=1, 
# Boundary conditions: u(x,0) = sin(pi*x), u(0,t) = 0, pi*e-t+du/dx(1,t) = 0
# Full orthogonal discretization method 

from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca 
from Collocation import *
# 1D Orthogonal Spline Collocation Method --> PDE to ODE 

# -----------------------------------------------------------------------------
# Collocation setup
# -----------------------------------------------------------------------------
nicp = 1        # Number of (intermediate) collocation points per control interval
nk = 3          # Control discretization 
tf = 2          # End time 
L = 1           # Space Domain
ndstate = 1     # State variables u
nastate = 0     # Algebraic state variable zC 
deg = 4 
cp = "radau"

B,C,D,tau_root,h,tau = coll_setup(nicp,nk,tf,ndstate,nastate,deg,cp)

# -----------------------------------------------------------------------------
# Model setup
# -----------------------------------------------------------------------------
# Constant parameters 
# State variables 
t = SX.sym("t")             # Time
z = SX.sym("z")             # Space 
n = nk*deg                  # Total Number of collocation points along the space domain 
x = SX.sym("x",n)           # Differential state (u) time discretization = n --> n differential states 
xa = SX.sym("xa",0)         # ALgebraic state 
xt = SX.sym("xt",n)         # Differential time derivative 
xd = SX.sym("xd",n)         # Differential state space derivative 
xdd = SX.sym("xdd",n)       # Differentail state space second derivative 
p = SX.sym("p",0)           # Symbolic parameter
u = SX.sym("u",0)           # Control Actions 

# du0/dt = du02/dx2     x [0,h0]
# du1/dt = d2u1/dx2     x [h0,h1]
rhs = vertcat(xdd)
ode  = rhs - np.pi**2*xdd 
# Algebraic constraint
alg = 0
# x0 = np.zeros(h)

# System dynamics (implicit formulation)
fcn = Function("fcn",[z,xdd,xt,x,xa,u,p],[ode])

# Algebraic equation (implicit formulation)
facn = Function("facn",[z,x,xa,u,p],[alg])

# Objective function (Mayer term)
mfcn = Function("mfcn",[z,x,xa,u,p],[])

# -- Recall the NLP for the orthogonal collocation --- 
V,g,vars_init,vars_lb,vars_ub,lbg,ubg,NP,ndiff,nu,nalg = problem_setup(t,x,xa,xd,xdd,u,p,nk,nicp,deg,h,B,C,D,fcn,facn,n)

# Space discretization using orthogonal collocation
z_collocation = np.linspace(0, L, n+1)  # Collocation points
z_sym = ca.MX.sym("z", n+1)  # Space nodes for differential states

# Differentiation matrix (Radau Collocation)
D_matrix = np.array(B)  # Obtain the Radau differentiation matrix from collocation setup

# Boundary conditions
u0 = ca.sin(np.pi * z_collocation)  # Initial condition at t = 0
boundary_left = 0  # u(0, t) = 0
boundary_right = pi * ca.exp(-t) + ca.MX.sym("dudx_at_1", 1)  # u'(1, t)

# PDE system
u = ca.MX.sym("u", n+1)  # States at collocation points
u_t = ca.MX.sym("u_t", n+1)  # Time derivatives at collocation points
u_xx = D_matrix @ u  # Second spatial derivative approximation
rhs = np.pi**2 * u_t - u_xx  # RHS of PDE

# Algebraic constraints and linking time integrator
dae = {
    "x": u,  # States
    "z": [],  # Algebraic variables (none in this case)
    "p": [],  # Parameters (if any)
    "ode": rhs,  # Time derivatives
    "alg": []  # Algebraic constraints
}

# Call CasADi integrator for the DAE
opts = {
    "grid": np.linspace(0, tf, nk),  # Time discretization
    "tf": tf,  # Total time for integration
}
integrator = ca.integrator("integrator", "idas", dae, opts)

# Solve the PDE
result = integrator(x0=u0)  # Initial conditions
u_opt = result["xf"]

# Visualize the results
u_results = np.array(u_opt).reshape(-1, n+1)
T, X = np.meshgrid(np.linspace(0, tf, nk), z_collocation)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, T, u_results.T, cmap="viridis", edgecolor="none")
ax.set_xlabel("Space (z)")
ax.set_ylabel("Time (t)")
ax.set_zlabel("u(z, t)")
ax.set_title("Solution of PDE with Orthogonal Collocation")
plt.show()
