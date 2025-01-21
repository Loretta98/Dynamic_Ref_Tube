from casadi import * 
import numpy as np 

def coll_setup(nicp,nk,L,ndstate,nastate,deg,cp,deg_t): 
    # Size of the finite element 
    h = L/nk/nicp

    # Coefficients of the collocation equation (1st order derivative)
    C = np.zeros((deg+1,deg+1))
    # Coefficients of the collocation equation (2nd order derivative)-
    B = np.zeros((deg+1,deg+1))
    # Coefficients of the collocation equation (1st order derivative in time)
    A = np.zeros((deg_t+1,deg_t+1))
    # Coefficients of the continuity equation
    D = np.zeros(deg+1)
    G = np.zeros(deg_t+1)
    # Collocation point
    tau = SX.sym("tau")
    # All collocation time points
    tau_root = [0] + collocation_points(deg, cp)    # collocation points for time domain 
    S = np.zeros((nk,deg+1))                        # space discretization over nk domain points given a deg polynomial grade for the function 
    for i in range(nk):
        for j in range(deg+1):
            S[i][j] = h*(i + tau_root[j])

    # For all collocation points: eq 10.4 in Biegler's book
    # Construct Lagrange polynomials to get the polynomial basis at the collocation point --> the lagrange polynomial approximates the differential equation on the finite element
    for j in range(deg+1):
        P = 1
        for j2 in range(deg+1):
            if j2 != j:
                P *= (tau-tau_root[j2])/(tau_root[j]-tau_root[j2])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation sum(Pj(1))
        lfcn = Function('lfcn', [tau],[P])
        D[j] = lfcn(1.0)

        # Evaluate the first derivative of the polynomial
        dP_dtau = jacobian(P, tau)
        tfcn = Function('tfcn', [tau], [dP_dtau])
        for j2 in range(deg+1):
            C[j][j2] = tfcn(tau_root[j2])

        # Evaluate the second derivative of the polynomial
        d2P_dtau2 = jacobian(dP_dtau, tau)
        tfcn2 = Function('tfcn2', [tau], [d2P_dtau2])
        for j2 in range(deg+1):
            B[j][j2] = tfcn2(tau_root[j2])

    return A,B,C,D, G,tau_root,h,tau,S

def problem_setup(t,x,xa,xd,xdd,u,p,nk,nicp,deg,h,B,C,D,fcn,facn,n,u0): 

        # Dimensions of the problem
    ndiff = x.nnz()         # number of differential states
    nalg  = xa.nnz()        # number of algebraic states
    nx = ndiff+nalg         # total number of states
    nu = u.nnz()            # number of controls
    NP = p.nnz()            # number of parameters
    # --- Initialization and Bounds over the collocation points ---
    # Control bounds 
    u_min = np.array([])
    u_max = np.array([])
    u_init = np.array((nk*nicp*(deg+1))) # Initialization of the control actions, even if mantained constants

    # Differential state bounds along domain and initial guess 
    xD_min =  np.ones(ndiff) * [1.0e-12]
    xD_max =  np.ones(ndiff) * [100.0]
    # Initial conditions for x=0, u(0,t)=0.0
    xDi_min = np.array(ndiff) * [0.0]
    xDi_max = np.array(ndiff) * [0.0]
    # Final conditions
    xDf_min = np.ones(ndiff) * [1.0e-12]
    xDf_max = np.ones(ndiff) * [100.0]
    # Initialization 
    #xD_init = np.array((nk*nicp*(deg+1)) * [[np.zeros(n)]]) # Initial conditions specified for every time interval 
    xD_init = np.ones((nk * nicp * (deg + 1), ndiff))*u0
    #xD_init = np.reshape(xD_init, (-1, ndiff))
    
    # Algebraic state bounds and initial guess
    xA_min =  np.ones(nalg) * [1.0e-12]
    xA_max =  np.ones(nalg) * [100.0]
    # xAi_min = np.array([0.0])
    # xAi_max = np.array([0.0])
    # xAf_min = np.array([1.0e-12])
    # xAf_max = np.array([100.0])
    xA_init = np.zeros((nk * nicp * (deg + 1), nalg))
    #xA_init = np.array((nk*nicp*(deg+1)))
    #xA_init = np.reshape(xA_init, (-1, ndiff))

    # Second derivative bounds
    xDD_min =  np.ones(ndiff) * [1.0e-12]
    xDD_max =  np.ones(ndiff) * [100.0]
    xDD_init = np.ones(ndiff) * 0.0

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
    icfcn = Function("icfcn",[t,x,xa,u,p],[ic])

    # Path constraint
    pc_min = np.array([])
    pc_max = np.array([])
    pc = SX()
    pcfcn = Function("pcfcn",[t,x,xa,u,p],[pc])

    # Final constraint
    fc_min = np.array([])
    fc_max = np.array([])
    fc = xd[-1]-np.exp(-t)                   # du/dx = sum d(Pj)dx x(t) 
    fcfcn = Function("fcfcn",[t,xd,x,xa,u,p],[fc])
    
    print ('Number of differential states: ', ndiff)
    print ('Number of algebraic states:    ', nalg)
    print ('Number of controls:            ', nu)
    print ('Number of parameters:          ', NP)

    # Total number of variables
    NXD = nicp*nk*(deg+1)*ndiff         # Collocated differential states (u0,u1,u2) variabile del sistema 
    NXA = nicp*nk*deg*nalg              # Collocated algebraic states
    NU  = nk*nu                         # Parametrized controls
    NXF = ndiff                         # Final state (only the differential states)
    NDD = nicp * nk * (deg + 1) * ndiff  # Second derivatives to explicitally recall them from V 
    NV = NXD + NDD+ NXA + NU + NXF + NP  # Total variables
    #NV  = NXD+NXA+NU+NXF+NP         # Total number of degrees of freedom

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
    vars_lb[offset:offset+NP]   = p_min
    vars_ub[offset:offset+NP]   = p_max
    offset += NP

    # --- Orthogonal Collocation over variables in the system ---
    # Define state variables and bounds
    XD = np.resize(np.array([], dtype=MX), (nk + 1, nicp, deg + 1))
    XA = np.resize(np.array([], dtype=MX), (nk, nicp, deg))
    XDD = np.resize(np.array([], dtype=MX), (nk, nicp, deg + 1))
    U = np.resize(np.array([], dtype=MX), nk)

    for k in range(nk):  
        # Collocated states
        for i in range(nicp):
            # Internal control points
            for j in range(deg+1):
                # Get the expression for the state vector
                XD[k][i][j] = V[offset:offset+ndiff]
                # Add second derivatives
                XDD[k][i][j] = V[offset:offset + ndiff]
                if j !=0:
                    XA[k][i][j-1] = V[offset+ndiff:offset+ndiff+nalg]
                # Add the initial condition
                index = (deg+1)*(nicp*k+i) + j
                if k==0 and j==0 and i==0:
                    vars_init[offset:offset+ndiff] = xD_init[index,:]
                    vars_lb[offset:offset+ndiff] = xDi_min
                    vars_ub[offset:offset+ndiff] = xDi_max                    
                    offset += ndiff

                    vars_init[offset:offset + ndiff] = xDD_init
                    vars_lb[offset:offset + ndiff] = xDD_min
                    vars_ub[offset:offset + ndiff] = xDD_max
                    offset += ndiff

                else:
                    if j!=0: # algebriac states are not calculated at the first collocation point
                        vars_init[offset:offset+ndiff+nalg] = np.append(xD_init[index,:],xA_init[index,:])
                        vars_lb[offset:offset+ndiff+nalg] = np.append(xD_min,xA_min)
                        vars_ub[offset:offset+ndiff+nalg] = np.append(xD_max,xA_max)
                        offset += 2*ndiff+nalg
                    else:
                        vars_init[offset:offset+ndiff] = xD_init[index,:]
                        vars_lb[offset:offset+ndiff] = xD_min
                        vars_ub[offset:offset+ndiff] = xD_max
                        offset += ndiff

                        vars_init[offset:offset + ndiff] = xDD_init
                        vars_lb[offset:offset + ndiff] = xDD_min
                        vars_ub[offset:offset + ndiff] = xDD_max
                        offset += ndiff
                        

        # Parametrized controls
        U[k] = V[offset:offset+nu]
        vars_lb[offset:offset+nu] = u_min
        vars_ub[offset:offset+nu] = u_max
        #vars_init[offset:offset+nu] = u_init[index,:]
        vars_init[offset:offset+nu] = u_init
        offset += nu

    # State at end of the spatial domain
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
                    xdd = B * x 
                 # Add second derivative constraints
                xdd_constraint = XDD[k][i][j] - xp_jk2 / h ** 2
                g += [xdd_constraint]
                lbg.append(np.zeros(ndiff))
                ubg.append(np.zeros(ndiff))

                # Add collocation equations to the NLP
                [fk] = fcn.call([0., XDD[k][i][j], XD[k][i][j], XA[k][i][j-1], U[k], P])
                
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
    [fck] = fcfcn.call([0., xp_jk[-1]/h, XD[k][i][j], XA[k][i][j-1], U[k], P])
    g += [fck]
    lbg.append(fc_min)
    ubg.append(fc_max)

    return V,g,vars_init,vars_lb,vars_ub,lbg,ubg,NP,ndiff,nu,nalg,NXD, NDD

def retrieve_solution(ndiff,deg,nicp,nk,nu,nalg,NP,v_opt,tau,tau_root,h): 
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
    return 