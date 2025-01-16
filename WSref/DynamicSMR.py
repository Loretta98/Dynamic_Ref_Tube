# Dynamic model with finite differences (implicit) expression of the spatial derivatives 
# The method of lines is applied to discretize over space and integrate over time 
# The following simplifications are applied: Tw,Tf = const., eta = 0.1, no axial dispersion 

from Properties import * 
import numpy as np 
from casadi import * 
from Input import * 

L = 2                          # Reactor lenght 
ncomp = 5                      # Number of components: CH4,CO,CO2,H2,H2O 
N = 5                          # Number of spatial points
z = np.linspace(0,L,N)
dz = z[1]-z[0]

# Symbolic variables 
t = SX.sym('t')
P = SX.sym('P',N)
#w = SX.sym('w',N*(ncomp+1))     # Total number of differential time variables, ncomp * N spatial points + 1 energy balance * N spatial points
wCH4 = SX.sym('wCH4',N)
wCO = SX.sym('wCO',N)
wCO2 = SX.sym('wCO2',N)
wH2 = SX.sym('wH2',N)
wH2O = SX.sym('wH2O',N)
T = SX.sym('T',N)

# PDE --> ODE 
#dwdt = SX.sym(N)               # Time components equations on frac. mass composition
dCH4dt = SX.zeros(N)
dCOdt = SX.zeros(N)
dCO2dt = SX.zeros(N)
dH2dt = SX.zeros(N) 
dH2Odt = SX.zeros(N) 
dTdt = SX.zeros(N)              # Time energy equation 
dTdz = SX.zeros(N)
#dTdz2 = SX.zeros(N)
dCH4dz = SX.zeros(N)
dCOdz = SX.zeros(N)
dCO2dz = SX.zeros(N)
dH2dz = SX.zeros(N) 
dH2Odz = SX.zeros(N)
dPdz = SX.zeros(N)
# We start with just one spatial term 
# CH4dz2 = SX.zeros(N)
# COdz2 = SX.zeros(N)
# CO2dz2 = SX.zeros(N)
# H2dz2 = SX.zeros(N) 
# H2Odz2 = SX.sym(N)

# Finite difference loop for spatial derivatives defined as symbolic variables
for i in range(1, N-1):
    # First derivatives
    dCH4dz[i] = (wCH4[i] - wCH4[i-1]) / dz
    dCOdz[i] = (wCO[i] - wCO[i-1]) / dz
    dCO2dz[i] = (wCO2[i] - wCO2[i-1]) / dz
    dH2dz[i] = (wH2[i] - wH2[i-1]) / dz
    dH2Odz[i] = (wH2O[i] - wH2O[i-1]) / dz
    dTdz[i] = (T[i] - T[i-1]) / dz  # If T is time-dependent
    dPdz[i] = (P[i]-P[i-1])/dz
    # # Second derivatives
    # dCH4dz2[i] = (wCH4[i+1] - 2*wCH4[i] + wCH4[i-1]) / dz**2
    # dCOdz2[i] = (wCO[i+1] - 2*wCO[i] + wCO[i-1]) / dz**2
    # dCO2dz2[i] = (wCO2[i+1] - 2*wCO2[i] + wCO2[i-1]) / dz**2
    # dH2dz2[i] = (wH2[i+1] - 2*wH2[i] + wH2[i-1]) / dz**2
    # dH2Odz2[i] = (wH2O[i+1] - 2*wH2O[i] + wH2O[i-1]) / dz**2
    #dTdz2[i] = (dTdt[i+1] - 2*dTdt[i] + dTdt[i-1]) / dz**2  # Second derivative for temperature


# Mixture massive Specific Heat calculation (Shomate Equation)                                                        # Enthalpy of the reaction at the gas temperature [J/mol]
        # CH4,          CO,             CO2,            H2,              H2O    
c1 = np.array([-0.7030298, 25.56759,  24.99735, 33.066178 , 30.09 ])
c2 = np.array([108.4773,   6.096130,  55.18696, -11.363417, 6.832514])/1000
c3 = np.array([-42.52157,  4.054656, -33.69137, 11.432816 ,  6.793435])/1e6
c4 = np.array([5.862788, -2.671301,   7.948387, -2.772874 ,  -2.534480])/1e9
c5 = np.array([0.678565,  0.131021,  -0.136638, -0.158558 , 0.082139])*1e6

Cp_mol = c1+c2*T+c3*T**2+c4*T**3+c5/T**2                        # Molar specific heat per component [J/molK]
Cp = Cp_mol/MW*1000                                             # Mass specific heat per component [J/kgK]
Cpmix = np.sum(Cp* w0 )                                        # Mass specific heat [J/kgK]

yi = (w0*m_gas)
Pi = P * yi 
# Pi deve avere dimensione N 

U,lambda_gas,DynVis = HeatTransfer(T,Tc,n_comp, MW, Pc, yi, Cpmix, RhoGas,dTube, Dp, Epsilon, e_w, u, dTube_out, lambda_s)
rj,kr = Kinetics(T,R, Pi, RhoC, Epsilon,N)
Deff = Diffusivity(R,T,P,yi, n_comp, MW, MWmix, e_s, tau)
#Eta = EffectivenessF(p_h,c_h,n_h,s_h,Dp,lambda_gas,Deff,kr)

######## Simplifications ############ 
Eta = 0.1
Tf = 850+273.15    # Constant value as first approximation

# Estimation of physical properties with ideal mixing rules
RhoGas = (Ppa*MWmix) / (R*T)  / 1000                                        # Gas mass density [kg/m3]
VolFlow_R1 = m_gas / RhoGas                                                 # Volumetric flow per tube [m3/s]
u = (F_R1*1000) * R * T / (Aint*Ppa)                                        # Superficial Gas velocity if the tube was empy (Coke at al. 2007)
u = VolFlow_R1 / (Aint * Epsilon)                                           # Gas velocity in the tube [m/s]
m_gas = m_R1
# rj should have size N 
rhs1 = -vz*dCH4dz   + vz*(Aint / (m_gas*3600) * MW[0] * np.sum(np.multiply(nu[:, 0], np.multiply(Eta, rj)))) 
rhs2 = -vz*dCOdz    + vz*(Aint / (m_gas*3600) * MW[1] * np.sum(np.multiply(nu[:, 1], np.multiply(Eta, rj))))
rhs3 = -vz*dCO2dz   + vz*(Aint / (m_gas*3600) * MW[2] * np.sum(np.multiply(nu[:, 2], np.multiply(Eta, rj))))
rhs4 = -vz*dH2dz    + vz*(Aint / (m_gas*3600) * MW[3] * np.sum(np.multiply(nu[:, 3], np.multiply(Eta, rj))))
rhs5 = -vz*dH2Odz   + vz*(Aint / (m_gas*3600) * MW[4] * np.sum(np.multiply(nu[:, 4], np.multiply(Eta, rj)))) 
rhs6 = (np.pi*dTube/(m_gas*Cpmix))*U*(Tf - T) -Aint/((m_gas*3600)*Cpmix)* (np.sum(np.multiply(DH_reaction, np.multiply(Eta,rj))))
# DynVis should have size N 
rhs7 = ( (-150 * (((1-Epsilon)**2)/(Epsilon**3)) * DynVis * u/ (Dp**2) - (1.75* ((1-Epsilon)/(Epsilon**3)) * m_gas*u/(Dp*Aint))  ) ) / 1e5

ode = vertcat(dCH4dt - rhs1, dCOdt - rhs2, dCO2dt - rhs3, dH2dt - rhs4, dH2Odt - rhs5, dTdt - rhs6)
alg = vertcat( dPdz-rhs7 )

dae = { 'wCH4': wCH4, 
        'wCO': wCO, 
        'wCO2': wCO2,
        'wH2': wH2, 
        'wH2O': wH2O,
        't': t, 
         'xa': P,
               }

# Define integrator with time grid
integrator = integrator('integrator', 'idas', dae, {'grid': time_points.tolist()})

print("Integration was succesfull")

# Add plots 