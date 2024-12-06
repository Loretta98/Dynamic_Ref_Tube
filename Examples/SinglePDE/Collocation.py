from casadi import * 
import numpy as np 

def coll_setup(nicp,nk,L,ndstate,nastate,deg,cp): 
    # Size of the finite element 
    h = L/nk/nicp

    # Coefficients of the collocation equation (1st order derivative)
    C = np.zeros((deg+1,deg+1))
    # Coefficients of the collocation equation (2nd order derivative)
    B = np.zeros((deg+1,deg+1))
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

        # # Evaluate the first time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        # tfcn = Function('tfcn', [tau],[tangent(L,tau)])
        # print(tfcn)
        # for j2 in range(deg+1):
        #     C[j][j2] = tfcn(tau_root[j2])
        
        # # Evaluate the second time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        # tfcn2 = Function('tfcn2', [tau], [tangent(tfcn,tau)])
        # for j2 in range(deg+1):
        #     B[j][j2] = tfcn2(tau_root[j2])

        # Evaluate the first derivative of the polynomial
        dL_dtau = jacobian(L, tau)
        tfcn = Function('tfcn', [tau], [dL_dtau])
        for j2 in range(deg+1):
            C[j][j2] = tfcn(tau_root[j2])

        # Evaluate the second derivative of the polynomial
        d2L_dtau2 = jacobian(dL_dtau, tau)
        tfcn2 = Function('tfcn2', [tau], [d2L_dtau2])
        for j2 in range(deg+1):
            B[j][j2] = tfcn2(tau_root[j2])

    return B,C,D,tau_root,h


def problem_setup(t,xd,xddot,xa,u,p,nk,nicp,deg,h,B,C,D,fcn,facn): 
        # Dimensions of the problem
    ndiff = xd.nnz()        # number of differential states
    nalg  = xa.nnz()        # number of algebraic states
    nx = ndiff+nalg          # total number of states
    nu = u.nnz()            # number of controls
    NP = p.nnz()            # number of parameters
    # --- Initialization and Bounds over the collocation points ---
    # Control bounds 
    u_min = np.array([])
    u_max = np.array([])
    u_init = np.array((nk*nicp*(deg+1))) # Initialization of the control actions, even if mantained constants

    # Differential state bounds along domain and initial guess 
    xD_min =  np.array([1.0e-12])
    xD_max =  np.array([100.0])
    # Initial conditions
    xDi_min = np.array([1.0])
    xDi_max = np.array([1.0])
    # Final conditions
    xDf_min = np.array([1.0e-12])
    xDf_max = np.array([100.0])
    # Initialization 
    xD_init = np.array((nk*nicp*(deg+1))*[0.0]) # Initial conditions specified for every time interval 
    xD_init = np.reshape(xD_init, (-1, ndiff))
    
    # Algebraic state bounds and initial guess
    xA_min =  np.array([1.0e-12])
    xA_max =  np.array([100.0])
    xAi_min = np.array([0.0])
    xAi_max = np.array([0.0])
    xAf_min = np.array([1.0e-12])
    xAf_max = np.array([100.0])
    xA_init = np.array((nk*nicp*(deg+1))*[0.0])
    xA_init = np.reshape(xA_init, (-1, ndiff))

    # Parameter bounds and initial guess
    p_min = np.array([])
    p_max = np.array([])
    p_init = np.array([])

    # -----------------------------------------------------------------------------
    # Constraints setup
    # -----------------------------------------------------------------------------

    # Initial constraint
    ic_min = np.array([0.0])
    ic_max = np.array([0.0])
    ic = SX()
    icfcn = Function("icfcn",[t,xd,xa,u,p],[ic])

    # Path constraint
    pc_min = np.array([])
    pc_max = np.array([])
    pc = SX()
    pcfcn = Function("pcfcn",[t,xd,xa,u,p],[pc])

    # Final constraint
    fc_min = np.array([])
    fc_max = np.array([])
    # fc = -np.exp(-t)
    fc = -np.exp(-t)                   # du/dx = sum d(Pj)dx x(t) 
    fcfcn = Function("fcfcn",[t,xd,xa,xddot,u,p],[xddot[-1]-fc])
    

    print ('Number of differential states: ', ndiff)
    print ('Number of algebraic states:    ', nalg)
    print ('Number of controls:            ', nu)
    print ('Number of parameters:          ', NP)

    # Total number of variables
    NXD = nicp*nk*(deg+1)*ndiff     # Collocated differential states
    NXDD = nicp*nk*(deg+1)*ndiff    # Collocated second differential states 
    NXA = nicp*nk*deg*nalg          # Collocated algebraic states
    NU  = nk*nu                     # Parametrized controls
    NXF = ndiff                     # Final state (only the differential states)
    NV  = NXD+NXA+NU+NXF+NP         # Total number of degrees of freedom

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
                # Get an expression for the state derivative at each collocation point
                xp_jk = 0
                xp_jk2 = 0 
                for j2 in range (deg+1):
                    xp_jk += C[j2][j]*XD[k][i][j2]          # get the time derivative of the differential states (eq 10.19b)
                    xp_jk2 += B[j2][j]*XD[k][i][j2]         # get the second time derivate of the differential states
                # Add collocation equations to the NLP
                [fk] = fcn.call([0., xp_jk/h, xp_jk2/h**2,XD[k][i][j], XA[k][i][j-1], U[k], P])
                g += [fk[:ndiff]] 
                # impose system dynamics (for the differential states (eq 10.19b))
                lbg.append(np.zeros(ndiff)) # equality constraints
                ubg.append(np.zeros(ndiff)) # equality constraints
                if nalg !=0:
                    [fak] = facn.call([0., XD[k][i][j], XA[k][i][j-1], U[k], P])
                    g += [fak[:nalg]]                              # impose system dynamics (for the algebraic states (eq 10.19b))
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

    # Final constraints (Const, dConst, ConstQ)
    [fck] = fcfcn.call([0., XD[k][i][j], XA[k][i][j-1], U[k], P])
    g += [fck]
    lbg.append(fc_min)
    ubg.append(fc_max)

    return V 