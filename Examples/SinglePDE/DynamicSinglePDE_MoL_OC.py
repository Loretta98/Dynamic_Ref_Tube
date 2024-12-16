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

# du0/dt = d2u0/dx2     x [0,h0]
# du1/dt = d2u1/dx2     x [h0,h1]
rhs = vertcat(xdd/np.pi**2)
# Algebraic constraint for the BC at x=1
alg = 0

# System dynamics (implicit formulation)
fcn = Function("fcn",[z,xdd,x,xa,u,p],[rhs])

# Algebraic equation (implicit formulation)
facn = Function("facn",[z,x,xa,u,p],[alg])

# Objective function (Mayer term)
mfcn = Function("mfcn",[z,x,xa,u,p],[])

# Set NLP solver options
opts = {}
opts["expand"] = True
opts["ipopt.max_iter"] = 1000 
opts["ipopt.tol"] = 1e-2
#opts["ipopt.linear_solver"] = 'ma27'

# The integrator is here recalled for integration of the DAE system 
space_domain = np.linspace(0,1,n)
u0 = np.ones(n)*np.sin(np.pi*space_domain)
time_points = np.linspace(0, tf, 100)  # Intermediate time points

# Correct Implementation of the BC 
u_results = []  # Store results for all time steps
current_u = u0 # Initial condition
Obj = 0 

for dt in time_points[1:]:
    # -- Recall the NLP for the orthogonal collocation --- 
    u_init = current_u
    V,g,vars_init,vars_lb,vars_ub,lbg,ubg,NP,ndiff,nu,nalg,NXD,NXDD = problem_setup(t,x,xa,xd,xdd,u,p,nk,nicp,deg,h,B,C,D,fcn,facn,n,u_init)
    ## ----
    ## SOLVE THE NLP
    ## ----
    nlp = {'x':V, 'f':Obj, 'g':vertcat(*g)}  
    # Allocate an NLP solver
    solver = nlpsol("solver", "ipopt", nlp, opts)
    arg = {
        "x0": vars_init,
        "lbx": vars_lb,
        "ubx": vars_ub,
        "lbg": np.concatenate(lbg),
        "ubg": np.concatenate(ubg),
    }
    # Solve the problem
    res = solver(**arg)
    print(f"nk={nk}, nicp={nicp}, deg+1={deg + 1}, ndiff={ndiff}")
    print(f"Expected size: {nk * nicp * (deg + 1) * ndiff}")
    # Retrieve the solution
    v_opt = np.array(res["x"])

    # Define offsets as calculated in `problem_setup`
    offset_xdd_start = ndiff
    offset_xdd_end = offset_xdd_start + NXDD  # End of `xdd`
    
    size_extracted = v_opt[offset_xdd_start:offset_xdd_end].size
    print(f"Extracted size: {size_extracted}")
    
    
    # Extract `xdd` from `v_opt`
    xdd_opt = v_opt[offset_xdd_start:offset_xdd_end].reshape((ndiff, nicp *(deg + 1)* nk))
    # Convert xdd_opt to a CasADi MX type
    #xdd_opt_casadi = SX(xdd_opt)  # Use SX or DM depending on your need
    
    xdd += B[j2][j]*x
    rhs = vertcat(xdd/np.pi**2) # array 
    ode  = xt - rhs
    dae = {
    'x': vertcat(x),  # Differential states
    'z': vertcat(),   # Algebraic state
    'p': vertcat(p),  # Parameters (symbolic)
    'ode': ode,
    'alg': alg
    }
    integrator = integrator('integrator','idas',dae, {'tf':dt})
    x0 = xd_opt
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