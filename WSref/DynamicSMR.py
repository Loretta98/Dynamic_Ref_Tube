# Dynamic model with finite differences (implicit) expression of the spatial derivatives 
# The method of lines is applied 
# The following simplifications are applied: Tw,Tf = const., eta = 0.1, no axial dispersion 

from Properties import * 
import numpy as np 
from casadi import * 

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
dH2Odz = SX.sym(N)

# We start with just one spatial term 
# CH4dz2 = SX.zeros(N)
# COdz2 = SX.zeros(N)
# CO2dz2 = SX.zeros(N)
# H2dz2 = SX.zeros(N) 
# H2Odz2 = SX.sym(N)

# dwdz = SX.sym(N)
# dwdz2 = SX.sym(N)

# Finite difference loop for spatial derivatives defined as symbolic variables
for i in range(1, N-1):
    # First derivatives
    dCH4dz[i] = (wCH4[i] - wCH4[i-1]) / dz
    dCOdz[i] = (wCO[i] - wCO[i-1]) / dz
    dCO2dz[i] = (wCO2[i] - wCO2[i-1]) / dz
    dH2dz[i] = (wH2[i] - wH2[i-1]) / dz
    dH2Odz[i] = (wH2O[i] - wH2O[i-1]) / dz

    dTdz[i] = (dTdt[i] - dTdt[i-1]) / dz  # If T is time-dependent

    # # Second derivatives
    # dCH4dz2[i] = (wCH4[i+1] - 2*wCH4[i] + wCH4[i-1]) / dz**2
    # dCOdz2[i] = (wCO[i+1] - 2*wCO[i] + wCO[i-1]) / dz**2
    # dCO2dz2[i] = (wCO2[i+1] - 2*wCO2[i] + wCO2[i-1]) / dz**2
    # dH2dz2[i] = (wH2[i+1] - 2*wH2[i] + wH2[i-1]) / dz**2
    # dH2Odz2[i] = (wH2O[i+1] - 2*wH2O[i] + wH2O[i-1]) / dz**2

    #dTdz2[i] = (dTdt[i+1] - 2*dTdt[i] + dTdt[i-1]) / dz**2  # Second derivative for temperature


rhs1 = 
rhs2 = 
rhs3 = 
rhs4 = 
rhs5 = 
rhs6 = 
rhs7 = 

ode = vertcat(dCH4dt - rhs1, dCOdt - rhs2, dCO2dt - rhs3, dH2dt - rhs4, dH2Odt - rhs5, dTdt - rhs6)

alg = vertcat(P-rhs7 )

dae = {}


# Define integrator with time grid
integrator = integrator('integrator', 'idas', dae, {'grid': time_points.tolist()})
