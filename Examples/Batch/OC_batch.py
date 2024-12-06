# @loret 
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
# Resolution with the orthogonal collocation method 

import numpy as np 
from casadi import * 
import matplotlib.pyplot as plt 

# -----------------------------------------------------------------------------
# Collocation setup
# -----------------------------------------------------------------------------
nicp = 1        # Number of (intermediate) collocation points per control interval
nk = 3          # Control discretization 
tf = 1          # End time 
ndstate = 2     # State variables zA,zB
nastate = 1     # Algebraic state variable zC 

# Degree of interpolating polynomial 
deg = 4 
# Radau collocation points 
cp = "radau"
# Size of the finite element 
h = tf/nk/nicp

# Coefficients of the collocation equation 
C = np.zeros((deg+1,deg+1))

# Coefficients of the continuity equation
D = np.zeros(deg+1)
# Collocation point
tau = SX.sym("tau")
# All collocation time points
tau_root = [0] + collocation_points(deg, cp)    # collocation points for time domain 
T = np.zeros((nk,deg+1))                        # time discretization over nk domain points given a deg polynomial grade for the function 
for i in range(nk):
    for j in range(deg+1):
        T[i][j] = h*(i + tau_root[j])

# For all collocation points: eq 10.4 in Biegler's book
# Construct Lagrange polynomials to get the polynomial basis at the collocation point --> the lagrange polynomial approximates the differential equation on the finite element
for j in range(deg+1):
    L = 1
    for j2 in range(deg+1):
        if j2 != j:
            L *= (tau-tau_root[j2])/(tau_root[j]-tau_root[j2])

    # Evaluate the polynomial at the final time to get the coefficients of the continuity equation sum(Pj(1))
    lfcn = Function('lfcn', [tau],[L])
    D[j] = lfcn(1.0)

    # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
    tfcn = Function('tfcn', [tau],[tangent(L,tau)])
    for j2 in range(deg+1):
        C[j][j2] = tfcn(tau_root[j2])


# -----------------------------------------------------------------------------
# Model setup
# -----------------------------------------------------------------------------
# Constant parameters 
p1 = 4.0
p2 = 2.0 
p3 = 40.0 
p4 = 20.0 
# State Variables 
# Time
t = SX.sym("t")             # time
xd = SX.sym("xd",2)         # Differential state (zA,zB)
xa = SX.sym("xa",1)         # Algebraic state (zC)
xddot = SX.sym("xdot",2)    # Differential state time derivative (dzA, dzB) 
p = SX.sym("p",0)           # Symbolic paramaters
u = SX.sym("u",0)           # Control actions
# ODE right hand side function 
rhs = vertcat(-p1*xd[0]+p2*xd[1], pi*xd[0]-(p2+p3)*xd[1]+p4*xa[0])

# AE right hand side function 
rhsalg = vertcat(1-xd[0]-xd[1])

# System dynamics (implicit formulation)
ffcn = Function("ffcn",[t,xddot,xd,xa,u,p],[xddot-rhs])

# Algebraic equation (implicit formulation)
fafcn = Function("facfcn",[t,xd,xa,u,p],[xa-rhsalg])

# Objective function (Mayer term)
mfcn = Function("mfcn",[t,xd,xa,u,p],[])

# --- Initialization and Bounds over the collocation points ---
# Control bounds 
u_min = np.array([])
u_max = np.array([])
u_init = np.array((nk*nicp*(deg+1))) # Initialization of the control actions, even if mantained constants

# Differential state bounds along domain and initial guess 
xD_min =  np.array([1.0e-12, 1.0e-12])
xD_max =  np.array([100.0, 100.0])
# Initial conditions
xDi_min = np.array([1.0 , 0.0])
xDi_max = np.array([1.0 , 0.0])
# Final conditions 
xDf_min = np.array([1.0e-12, 1.0e-12])
xDf_max = np.array([100.0 ,100.0])
# Initialization 
xD_init = np.array((nk*nicp*(deg+1))*[[1.0 , 0.0 ]]) # Initial conditions specified for every time interval 

# Algebraic state bounds and initial guess
xA_min =  np.array([1.0e-12])
xA_max =  np.array([100.0])
xAi_min = np.array([0.0])
xAi_max = np.array([0.0])
xAf_min = np.array([1.0e-12])
xAf_max = np.array([100.0])
xA_init = np.array((nk*nicp*(deg+1))*[[0.0]])

# Parameter bounds and initial guess
p_min = np.array([])
p_max = np.array([])
p_init = np.array([])

# Initialize functions
#ffcn.init()
#fafcn.init()
#mfcn.init()

# -----------------------------------------------------------------------------
# Constraints setup
# -----------------------------------------------------------------------------

# Initial constraint
ic_min = np.array([])
ic_max = np.array([])
ic = SX()
#ic.append();       ic_min = append(ic_min, 0.);         ic_max = append(ic_max, 0.)
icfcn = Function("icfcn",[t,xd,xa,u,p],[ic])

# Path constraint
pc_min = np.array([])
pc_max = np.array([])
pc = SX()
#pc.append();       pc_min = append(pc_min, 0.);         pc_max = append(pc_max, 0.)
pcfcn = Function("pcfcn",[t,xd,xa,u,p],[pc])

# Final constraint
fc_min = np.array([])
fc_max = np.array([])
fc = SX()
#fc.append();       fc_min = append(fc_min, 0.);         fc_max = append(fc_max, 0.)
fcfcn = Function("fcfcn",[t,xd,xa,u,p],[fc])

# Initialize the functions
#icfcn.init()
#pcfcn.init()
#fcfcn.init()

# -----------------------------------------------------------------------------
# NLP setup
# -----------------------------------------------------------------------------
# Dimensions of the problem
ndiff = xd.nnz()        # number of differential states
nalg  = xa.nnz()        # number of algebraic states
nx = ndiff+nalg          # total number of states
nu = u.nnz()            # number of controls
NP = p.nnz()            # number of parameters

print ('Number of differential states: ', ndiff)
print ('Number of algebraic states:    ', nalg)
print ('Number of controls:            ', nu)
print ('Number of parameters:          ', NP)

# Total number of variables
NXD = nicp*nk*(deg+1)*ndiff # Collocated differential states
NXA = nicp*nk*deg*nalg      # Collocated algebraic states
NU  = nk*nu                 # Parametrized controls
NXF = ndiff                 # Final state (only the differential states)
NV  = NXD+NXA+NU+NXF+NP     # Total number of degrees of freedom

# NLP variable vector
V = MX.sym("V",NV)
  
# All variables with bounds and initial guess
vars_lb = np.zeros(NV)
vars_ub = np.zeros(NV)
vars_init = np.zeros(NV)
offset = 0

# Get the parameters
P = V[offset:offset+NP]
vars_init[offset:offset+NP] = p_init
vars_lb[offset:offset+NP] = p_min
vars_ub[offset:offset+NP] = p_max
offset += NP

# --- Orthogonal Collocation over variables in the system --- 
# Get collocated states and parametrized control
XD = np.resize(np.array([],dtype=MX),(nk+1,nicp,deg+1))  # NB: same name as above
XA = np.resize(np.array([],dtype=MX),(nk,nicp,deg))      # NB: same name as above
U  = np.resize(np.array([],dtype=MX),nk)
for k in range(nk):  
    # Collocated states
    for i in range(nicp):
        # Internal control points
        for j in range(deg+1):
            # Get the expression for the state vector
            XD[k][i][j] = V[offset:offset+ndiff]
            if j !=0:
                XA[k][i][j-1] = V[offset+ndiff:offset+ndiff+nalg]
            # Add the initial condition
            index = (deg+1)*(nicp*k+i) + j
            if k==0 and j==0 and i==0:
                vars_init[offset:offset+ndiff] = xD_init[index,:]
                vars_lb[offset:offset+ndiff] = xDi_min
                vars_ub[offset:offset+ndiff] = xDi_max                    
                offset += ndiff
            else:
                if j!=0: # algebriac states are not calculated at the first collocation point
                    vars_init[offset:offset+ndiff+nalg] = np.append(xD_init[index,:],xA_init[index,:])
                    vars_lb[offset:offset+ndiff+nalg] = np.append(xD_min,xA_min)
                    vars_ub[offset:offset+ndiff+nalg] = np.append(xD_max,xA_max)
                    offset += ndiff+nalg
                else:
                    vars_init[offset:offset+ndiff] = xD_init[index,:]
                    vars_lb[offset:offset+ndiff] = xD_min
                    vars_ub[offset:offset+ndiff] = xD_max
                    offset += ndiff

    # Parametrized controls
    U[k] = V[offset:offset+nu]
    vars_lb[offset:offset+nu] = u_min
    vars_ub[offset:offset+nu] = u_max
    #vars_init[offset:offset+nu] = u_init[index,:]
    vars_init[offset:offset+nu] = u_init
    offset += nu

# State at end time
XD[nk][0][0] = V[offset:offset+ndiff]
vars_lb[offset:offset+ndiff] = xDf_min
vars_ub[offset:offset+ndiff] = xDf_max
vars_init[offset:offset+ndiff] = xD_init[-1,:]
offset += ndiff
assert(offset==NV)

# Constraint function for the NLP
g   = []
lbg = []
ubg = []

# Initial constraints
[ick] = icfcn.call([0., XD[0][0][0], XA[0][0][0], U[0], P])
g += [ick]
lbg.append(ic_min)
ubg.append(ic_max)

# For all finite elements
for k in range(nk):
    for i in range(nicp):
        # For all collocation points
        for j in range(1,deg+1):   		
            # Get an expression for the state derivative at the collocation point
            xp_jk = 0
            for j2 in range (deg+1):
                xp_jk += C[j2][j]*XD[k][i][j2]       # get the time derivative of the differential states (eq 10.19b)
            
            # Add collocation equations to the NLP
            [fk] = ffcn.call([0., xp_jk/h, XD[k][i][j], XA[k][i][j-1], U[k], P])
            g += [fk[:ndiff]] 
                    # impose system dynamics (for the differential states (eq 10.19b))
            lbg.append(np.zeros(ndiff)) # equality constraints
            ubg.append(np.zeros(ndiff)) # equality constraints
            if nalg !=0:
                [fak] = fafcn.call([0., XD[k][i][j], XA[k][i][j-1], U[k], P])
                g += [fak[:nalg]]                               # impose system dynamics (for the algebraic states (eq 10.19b))
                lbg.append(np.zeros(nalg)) # equality constraints
                ubg.append(np.zeros(nalg)) # equality constraints
            
            #  Evaluate the path constraint function
            [pck] = pcfcn.call([0., XD[k][i][j], XA[k][i][j-1], U[k], P])
            g += [pck]
            lbg.append(pc_min)
            ubg.append(pc_max)
        
        # Get an expression for the state at the end of the finite element
        xf_k = 0
        for j in range(deg+1):
            xf_k += D[j]*XD[k][i][j]
            
        # Add continuity equation to NLP
        if i==nicp-1:
#            print "a ", k, i
            g += [XD[k+1][0][0] - xf_k]
        else:
#            print "b ", k, i
            g += [XD[k][i+1][0] - xf_k]
        
        lbg.append(np.zeros(ndiff))
        ubg.append(np.zeros(ndiff))

# Periodicity constraints 


# Final constraints (Const, dConst, ConstQ)
[fck] = fcfcn.call([0., XD[k][i][j], XA[k][i][j-1], U[k], P])
g += [fck]
lbg.append(fc_min)
ubg.append(fc_max)


# Nonlinear constraint function
#gfcn = MXFunction([V],[vertcat(g)])

# Objective function of the NLP
# Regularization
Obj = 0
#for k in range(nk):
#    for i in range(nicp):
#        # For all collocation points
#        for j in range(1,deg+1):
#            [obj] = mfcn2.call([XD[k][i][j],U[k], P])
#            Obj += obj
#[obj] = mfcn2.call([XD[nk][0][0],zeros(9), P])
#Obj += obj
# Energy
# obj = mfcn(0., XD[k][i][j], XA[k][i][j-1], U[k], P)
# Obj += obj

#ofcn = MXFunction([V], [Obj])


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

# Solve the problem
res = solver(**arg)

# Print the optimal cost
print("optimal cost: ", float(res["f"]))

# Retrieve the solution
v_opt = np.array(res["x"])

# Print the optimal cost
print ("optimal cost: ", float(res["f"]))

# Retrieve the solution
v_opt = np.array(res["x"])
    

## ----
## RETRIEVE THE SOLUTION
## ---- 
xD_opt = np.resize(np.array([],dtype=MX),(ndiff,(deg+1)*nicp*(nk)+1))
xA_opt = np.resize(np.array([],dtype=MX),(nalg,(deg)*nicp*(nk)))
u_opt  = np.resize(np.array([],dtype=MX),(nu,(deg+1)*nicp*(nk)+1))
p_opt  = np.resize(np.array([],dtype=MX),(NP,1))

offset = 0
offset2 = 0
offset3 = 0
offset4 = 0

# Retrieve the parameter
p_opt = v_opt[:NP][:,0]
offset += NP    

# Retrieve differential, algebraic and control variables
for k in range(nk):  
    for i in range(nicp):
        for j in range(deg+1):
            xD_opt[:,offset2] = v_opt[offset:offset+ndiff][:,0]
            offset2 += 1
            offset += ndiff
            if j!=0:
                xA_opt[:,offset4] = v_opt[offset:offset+nalg][:,0]
                offset4 += 1
                offset += nalg
    utemp = v_opt[offset:offset+nu][:,0]
    for i in range(nicp):
        for j in range(deg+1):
            u_opt[:,offset3] = utemp
            offset3 += 1
    #    u_opt += v_opt[offset:offset+nu]
    offset += nu
    
xD_opt[:,-1] = v_opt[offset:offset+ndiff][:,0]    


    
# The algebraic states are not defined at the first collocation point of the finite elements:
# with the polynomials we compute them at that point
Da = np.zeros(deg)
for j in range(1,deg+1):
    # Lagrange polynomials for the algebraic states: exclude the first point
    La = 1
    for j2 in range(1,deg+1):
        if j2 != j:
            La *= (tau-tau_root[j2])/(tau_root[j]-tau_root[j2])
    lafcn = Function("lafcn", [tau], [La])
   # lafcn.init()
   # lafcn.setInput(tau_root[0])
    #lafcn.evaluate()
    Da[j-1] = lafcn(tau_root[0])

xA_plt = np.resize(np.array([],dtype=MX),(nalg,(deg+1)*nicp*(nk)+1))
offset4=0
offset5=0
for k in range(nk):  
    for i in range(nicp):
        for j in range(deg+1):
            if j!=0:         
                xA_plt[:,offset5] = xA_opt[:,offset4]
                offset4 += 1
                offset5 += 1
            else:
                xa0 = 0
                for j in range(deg):
                    xa0 += Da[j]*xA_opt[:,offset4+j]
                xA_plt[:,offset5] = xa0
                #xA_plt[:,offset5] = xA_opt[:,offset4]
                offset5 += 1

xA_plt[:,-1] = xA_plt[:,-2]    
    
# Construct the time grid    
tg = np.array(tau_root)*h
for k in range(nk*nicp):
    if k == 0:
        tgrid = tg
    else:
        tgrid = np.append(tgrid,tgrid[-1]+tg)
tgrid = np.append(tgrid,tgrid[-1])

# Print parameter
if NP!=0:
    print ('optimal parameter: ', float(p_opt))

#for k in range(nk*nicp):
print ('xd(0) =', xD_opt[0,:] )

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(tgrid, xD_opt[0,:], label="zA", marker="o")
plt.plot(tgrid, xD_opt[1,:], label="zB", marker="s")
plt.plot(tgrid, xA_plt[0,:], label="zC", marker="^")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Concentration Evolution in Batch Reactor")
plt.legend()
plt.grid()
plt.show()