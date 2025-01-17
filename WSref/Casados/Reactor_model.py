from casadi import * 
from Input import * 
from Properties_N import * 
from acados_template import AcadosModel

from Properties import Kinetics

def create_reactor_dynamics(N,L): 
    model_name = "reactor_model"
    ncomp=5
    # Constants
    Aint = 0.1
    m_gas = 1.0
    Cpmix = 2250
    U = 60
    Tf = 850 + 273.15  # Convert to Kelvin
    dTube = 0.1
    pi = np.pi
    MW = DM([16.04, 28.01, 44.01, 2.016, 18.015])
    const = Aint / (m_gas * 3600)  # Flow rate adjustment

    dCH4dt = SX.sym('dCH4dt',N)
    dCOdt = SX.sym('dCOdt',N)
    dCO2dt = SX.sym('dCO2dt',N)
    dH2dt = SX.sym('dH2dt',N) 
    dH2Odt = SX.sym('dH2Odt',N) 
    dTdt = SX.sym('dTdt',N)              # Time energy equation 
    
    # State variables
    t = SX.sym('t')
    wCH4 = SX.sym('wCH4', N)
    wCO = SX.sym('wCO', N)
    wCO2 = SX.sym('wCO2', N)
    wH2 = SX.sym('wH2', N)
    wH2O = SX.sym('wH2O', N)
    T = SX.sym('T', N)
    z = SX.sym('z',1)
    u = SX.sym('u',1)

    omega = vertcat(wCH4,wCO,wCO2,wH2,wH2O)
    # Derivatives
    dCH4dz = SX.zeros(N)
    dCOdz = SX.zeros(N)
    dCO2dz = SX.zeros(N)
    dH2dz = SX.zeros(N) 
    dH2Odz = SX.zeros(N)
    dTdz=SX.zeros(N)

    Z = np.linspace(0,L,N)
    dz = Z[1]-Z[0]
    for i in range(1, N-1):
    # First derivatives
        dCH4dz[i] = (wCH4[i] - wCH4[i-1]) / dz
        dCOdz[i] = (wCO[i] - wCO[i-1]) / dz
        dCO2dz[i] = (wCO2[i] - wCO2[i-1]) / dz
        dH2dz[i] = (wH2[i] - wH2[i-1]) / dz
        dH2Odz[i] = (wH2O[i] - wH2O[i-1]) / dz
        dTdz[i] = (T[i] - T[i-1]) / dz 

    P = DM.zeros(N) + np.ones(N)*Pin_R1                  # Constant pressure 
    yi = SX.zeros((N, ncomp))
    Pi = SX.zeros((N, ncomp))
    MW = DM([16.04, 28.01, 44.01, 2.016, 18.015]) 
    # Loop over spatial points
    for i in range(0,N):
        # Calculate mass fractions at each spatial point
        omega_at_i = omega[i::N]  # Extract omega for spatial point i
        # Compute yi (mole fractions)
        molar_fractions = (m_gas * omega_at_i) / MW  # Element-wise division
        total_moles = sum1(molar_fractions)         # Sum over components (scalar)
        yi[i,:] = molar_fractions / total_moles    # Normalize to get mole fractions at each element
        # Compute Pi (partial pressures)
        Pi[i,:] = P[i] * yi[i::N]
    Ppa = P*1e-5
    # Estimation of physical properties with ideal mixing rules
    RhoGas = (Ppa*MWmix) / (R*T)  / 1000                                            # Gas mass density [kg/m3]
    VolFlow_R1 = m_gas / RhoGas                                                     # Volumetric flow per tube [m3/s]
    u_ = (F3) * R * T / (Aint*Ppa)                                                   # Superficial Gas velocity if the tube was empy (Coke at al. 2007)
    vz  = VolFlow_R1 / (Aint * Epsilon)                                               # Gas velocity in the tube [m/s]
    rj,kr = Kinetics_N(T,R, Pi, RhoC, Epsilon,N)
    Dh_reaction = SX.zeros(3,N) + [225054923.824, -35038982.22,190015941.6]
    Eta = SX.zeros(3,N) + 0.1

    # RHS expressions
    rhs1 = -vz * dCH4dz + (vz.T * (const * MW[0] * sum1(nu_[:, 0] * (Eta * rj)))).T
    rhs2 = -vz * dCOdz + (vz.T * (const * MW[1] * sum1(nu_[:, 1] * (Eta * rj)))).T
    rhs3 = -vz * dCO2dz + (vz.T * (const * MW[2] * sum1(nu_[:, 2] * (Eta * rj)))).T
    rhs4 = -vz * dH2dz + (vz.T * (const * MW[3] * sum1(nu_[:, 3] * (Eta * rj)))).T
    rhs5 = -vz * dH2Odz + (vz.T * (const * MW[4] * sum1(nu_[:, 4] * (Eta * rj)))).T
    rhs6 = (pi * dTube / (m_gas * Cpmix)) * U * (Tf - T) - const / Cpmix * (sum1(Dh_reaction * (Eta * rj))).T

    # Combine RHS
    xdot = vertcat(dCH4dt,dCOdt,dCO2dt,dH2dt,dH2Odt,dTdt)
    f_expl = vertcat(rhs1, rhs2, rhs3, rhs4, rhs5, rhs6)
    ode = vertcat(dCH4dt - rhs1, dCOdt - rhs2, dCO2dt - rhs3, dH2dt - rhs4, dH2Odt - rhs5, dTdt - rhs6)
    alg = vertcat()
    #alg = vertcat( dPdz-rhs7 )
    x = vertcat(wCH4, wCO, wCO2, wH2, wH2O, T)
    z = vertcat()
    u = vertcat()
    # Define the DAE system correctly
    dae = Function('reactor_dae', [xdot,x,z,u], [ode,alg])  # Differential stat

    model = AcadosModel()
    model.f_impl_expr = ode
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = z
    #model.p = p

    model.name = model_name
    return model 

# Define the wall temperature function
def Twall(z, a, b, c):
    return a * (1 - np.exp(-b * z)) + c

b = 2  # Adjusted for the curve shape
c = 600+273.15  # Starting temperature

def create_reactor_steady(N,L): 
    model_name = "reactor_model_SS"
    ncomp=5
    # Constant
    U = 60
    #Tf = 850 + 273.15  # Convert to Kelvin
    T = 600+273.15
    dTube = 0.1
    pi = np.pi
    MW = DM([16.04, 28.01, 44.01, 2.016, 18.015])
    const = Aint / (m_gas * 3600)  # Flow rate adjustment

    dCH4dz = SX.sym('dCH4dz')
    dCOdz = SX.sym('dCOdz')
    dCO2dz = SX.sym('dCO2dz')
    dH2dz = SX.sym('dH2dz') 
    dH2Odz = SX.sym('dH2Odz') 
    dTdz = SX.sym('dTdz')              # Time energy equation 
    
    # State variables
    #t = SX.sym('t')
    wCH4 = SX.sym('wCH4')
    wCO = SX.sym('wCO')
    wCO2 = SX.sym('wCO2')
    wH2 = SX.sym('wH2')
    wH2O = SX.sym('wH2O')
    #T = SX.sym('T')
    z = SX.sym('z')
    u = SX.sym('u')

    omega = vertcat(wCH4,wCO,wCO2,wH2,wH2O)
        # Mixture massive Specific Heat calculation (Shomate Equation)                                                        # Enthalpy of the reaction at the gas temperature [J/mol]
            # CH4,          CO,             CO2,            H2,              H2O    
    c1 = np.array([-0.7030298, 25.56759,  24.99735, 33.066178 , 30.09 ])
    c2 = np.array([108.4773,   6.096130,  55.18696, -11.363417, 6.832514])/1000
    c3 = np.array([-42.52157,  4.054656, -33.69137, 11.432816 ,  6.793435])/1e6
    c4 = np.array([5.862788, -2.671301,   7.948387, -2.772874 ,  -2.534480])/1e9
    c5 = np.array([0.678565,  0.131021,  -0.136638, -0.158558 , 0.082139])*1e6
 
    Cp_mol = c1+c2*T+c3*T**2+c4*T**3+c5/T**2                        # Molar specific heat per component [J/molK]
    Cp = Cp_mol/MW*1000                                             # Mass specific heat per component [J/kgK]
    Cpmix = sum1(Cp*omega)   
    # Dh_reaction = SX.zeros(3)
    # for i in range(0,3):                                     # Mass specific heat [J/kgK]
    #     Dh_reaction[i] = DHreact[i]*1000 + sum1(nu_[i]*(c1*(T-298) + c2*(T**2-298**2)/2 + c3*(T**3-298**3)/3 + c4*(T**4-298**4)/4 - c5*(1/T-1/298))) #J/mol
    # Dh_reaction = Dh_reaction*1000 #J/kmol
    # print(Cpmix,Dh_reaction)
    Z = np.linspace(0,L,N)
    #P = DM.zeros(N) + np.ones(N)*Pin_R1                  # Constant pressure 
    P = Pin_R1
    yi = SX.zeros((N, ncomp))
    Pi = SX.zeros((N, ncomp))
    MW = DM([16.04, 28.01, 44.01, 2.016, 18.015]) 
    # Loop over spatial points
    a = Tf-Twin  # (900°C - 600°C)
    Tw = Twall(z,a,b,Twin)

    mi = m_gas*omega                                        # Mass flowrate per tube per component [kg/s tube]
    ni = np.divide(mi,MW)                                   # Molar flowrate per tube per component [kmol/s tube]
    ntot = sum1(ni)                                       # Molar flowrate per tube [kmol/s tube]
    yi = ni/ntot                                            # Molar fraction

    Pi = P*yi                                               # Partial Pressure
    Ppa = P * 1E5                                           # Pressure [Pa]
    #Eta = 0.1                                               # effectiveness factor (Latham et al., Kumar et al.)
    # Estimation of physical properties with ideal mixing rules
    RhoGas = (Ppa*MWmix) / (R*T)  / 1000                                            # Gas mass density [kg/m3]
    VolFlow_R1 = m_gas / RhoGas                                                     # Volumetric flow per tube [m3/s]
    u_ = (F3) * R * T / (Aint*Ppa)                                                   # Superficial Gas velocity if the tube was empy (Coke at al. 2007)
    vz  = VolFlow_R1 / (Aint * Epsilon)                                               # Gas velocity in the tube [m/s]
    rj,kr = Kinetics(T,R, Pi, RhoC, Epsilon)
    #Dh_reaction = SX.zeros(3) + [225054923.824, -35038982.22,190015941.6]
    #rj = np.zeros(3) + [875248.66, 0.00104, 10147109677.00]
    Eta = SX.zeros(3) + 0.1

    # RHS expressions
    rhs1 =  ((const * MW[0] * sum1(nu_[:, 0] * (Eta * rj))))
    rhs2 =  ((const * MW[1] * sum1(nu_[:, 1] * (Eta * rj))))
    rhs3 = ((const * MW[2] * sum1(nu_[:, 2] * (Eta * rj))))
    rhs4 = ((const * MW[3] * sum1(nu_[:, 3] * (Eta * rj))))
    rhs5 = ((const * MW[4] * sum1(nu_[:, 4] * (Eta * rj))))
    #rhs6 = (pi * dTube / (m_gas * Cpmix)) * U * (Tf - T) - const / Cpmix * (sum1(Dh_reaction * (Eta * rj)))

    # Combine RHS
    xdot = vertcat(dCH4dz,dCOdz,dCO2dz,dH2dz,dH2Odz)#,dTdz)
    f_expl = vertcat(rhs1, rhs2, rhs3, rhs4, rhs5)#, rhs6)
    ode = vertcat(dCH4dz - rhs1, dCOdz - rhs2, dCO2dz - rhs3, dH2dz - rhs4, dH2Odz - rhs5)#, dTdz - rhs6)
    alg = vertcat()
    #alg = vertcat( dPdz-rhs7 )
    x = vertcat(wCH4, wCO, wCO2, wH2, wH2O)#, T)
    z = vertcat()
    u = vertcat()
    # # Define the DAE system correctly
    # dae = Function('reactor_dae', [xdot,x,z,u], [ode,alg])  # Differential stat

    model = AcadosModel()
    model.f_impl_expr = ode
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = z
    #model.p = p

    model.name = model_name
    return model 