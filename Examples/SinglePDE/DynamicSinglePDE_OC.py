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

B,C,D,tau_root,h = coll_setup(nicp,nk,tf,ndstate,nastate,deg,cp)

# -----------------------------------------------------------------------------
# Model setup
# -----------------------------------------------------------------------------
# Constant parameters 
# State variables 
t = SX.sym("t")             # Time
z = SX.sym("z")             # Space 
n = nk*deg
xd = SX.sym("xd",n)        # Differential state (u) time discretization = h --> h differential states 
xa = SX.sym("xa",0)         # ALgebraic state 
xdot = SX.sym("xdot",n)     # Differential time derivative 
xddot = SX.sym("xdot",n)    # Differential state space derivative 
xdddot = SX.sym("xdddot",n) # Differentail state space second derivative 
p = SX.sym("p",0)           # Symbolic parameter
u = SX.sym("u",0)           # Control Actions 


# du0/dt = du02/dx2     x [0,h0]
# du1/dt = d2u1/dx2     x [h0,h1]
rhs = vertcat(xdddot)
ode  = rhs - np.pi**2*xdot 
# Algebraic constraint
alg = 0
# x0 = np.zeros(h)

# System dynamics (implicit formulation)
fcn = Function("ffcn",[z,xddot,xd,xa,u,p],[ode])

# Algebraic equation (implicit formulation)
facn = Function("facfcn",[z,xd,xa,u,p],[alg])

# Objective function (Mayer term)
mfcn = Function("mfcn",[z,xd,xa,u,p],[])

# -- Recall the NLP for the orthogonal collocation --- 
V = problem_setup(t,xd,xddot,xa,u,p,nk,nicp,deg,h,B,C,D,fcn,facn)

# Solve the problem
res = solver(**arg)
xddot = ()
# --- Model Integration ---
## ----
## SOLVE THE NLP
## ----
nlp = {'x':V, 'f':Obj, 'g':vertcat(*g)}  
#assert(1==0)
# Allocate an NLP solver
#solver = IpoptSolver(ofcn,gfcn)

# Set options
opts = {}
opts["expand"] = True
opts["ipopt.max_iter"] = 1000
opts["ipopt.tol"] = 1e-2
#opts["ipopt.linear_solver"] = 'ma27'


# Allocate an NLP solver
solver = nlpsol("solver", "ipopt", nlp, opts)
arg = {}

# Initial condition
arg["x0"] = vars_init

# Bounds on x
arg["lbx"] = vars_lb
arg["ubx"] = vars_ub

# Bounds on g
arg["lbg"] = np.concatenate(lbg)
arg["ubg"] = np.concatenate(ubg)


# Define DAE system
dae = {
    'x': vertcat(xd),  # Differential states
    'z': [],              # Algebraic state
    'p': vertcat(p),  # Parameters (symbolic)
    'ode': ode,
    'alg': alg
}

# Time grid for output
grid = np.linspace(0, 1, 5)  # Time grid from t0=0 to tf=1
t0 = 0
u0 = np.arrray(sol_ss)
# Create integrator
I = integrator('I', 'idas', dae, {'t0':t0, 'grid': grid})

# Initial conditions t=0 
u0 = 0
z0 = [0]
u_results = I(x0=u0, z0=z0, p=p)
print(u_results)
# u_results = np.array(u_results)  # Convert list of arrays to 2D NumPy array
# t_values = np.linspace(0,2)
# # Ensure `X`, `T`, and `u_results` are compatible
# X, T = np.meshgrid(x, t_values[:len(u_results)])  # Match time steps
# # 3D Surface Plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# surface = ax.plot_surface(X, T, u_results, cmap='viridis', edgecolor='none')
# ax.set_xlabel('x')
# ax.set_ylabel('Time (t)')
# ax.set_zlabel('u(x, t)')
# ax.set_title('Solution of the PDE over time (Orthogonal Collocation)')
# fig.colorbar(surface, label='u(x, t)')
# plt.show()