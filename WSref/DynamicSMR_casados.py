# Dynamic model with finite differences (implicit) expression of the spatial derivatives 
# The method of lines is applied to discretize over space and integrate over time 
# The following simplifications are applied: Tw,Tf = const., eta = 0.1, no axial dispersion 

# For more robust integration of the model within the NLP problem, the authors are implementing 
# acados wrapped through casados integrators to be handled by the NLP problem. 

from Properties import * 
import numpy as np 
from casadi import * 
from Input import * 
import matplotlib.pyplot as plt 
from casados_integrator import * 
from acados_simulator import create_awe_casados_integrator
import casadi as ca 
import time 
import pickle

# check that it is a 1-index DAE (now it is only an ODE)

L = 2                          # Reactor lenght 
ncomp = 5                      # Number of components: CH4,CO,CO2,H2,H2O 
N = 10                          # Number of spatial points
z = np.linspace(0,L,N)
dz = z[1]-z[0]

# Symbolic variables 
t = SX.sym('t')
#P = SX.sym('P',N)
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

# The boundaries conditions are here (gradient = 0)
dCH4dz = SX.zeros(N)
dCOdz = SX.zeros(N)
dCO2dz = SX.zeros(N)
dH2dz = SX.zeros(N) 
dH2Odz = SX.zeros(N)
dTdz=SX.zeros(N)
#dPdz = SX.zeros(N)

# Finite difference loop for spatial derivatives defined as symbolic variables
for i in range(1, N-1):
    # First derivatives
    dCH4dz[i] = (wCH4[i] - wCH4[i-1]) / dz
    dCOdz[i] = (wCO[i] - wCO[i-1]) / dz
    dCO2dz[i] = (wCO2[i] - wCO2[i-1]) / dz
    dH2dz[i] = (wH2[i] - wH2[i-1]) / dz
    dH2Odz[i] = (wH2O[i] - wH2O[i-1]) / dz
    dTdz[i] = (T[i] - T[i-1]) / dz  # If T is time-dependent
    #dPdz[i] = (P[i]-P[i-1])/dz

# # Mixture massive Specific Heat calculation (Shomate Equation)                                                        # Enthalpy of the reaction at the gas temperature [J/mol]
#         # CH4,          CO,             CO2,            H2,              H2O    
# c1 = np.array([-0.7030298, 25.56759,  24.99735, 33.066178 , 30.09 ])
# c2 = np.array([108.4773,   6.096130,  55.18696, -11.363417, 6.832514])/1000
# c3 = np.array([-42.52157,  4.054656, -33.69137, 11.432816 ,  6.793435])/1e6
# c4 = np.array([5.862788, -2.671301,   7.948387, -2.772874 ,  -2.534480])/1e9
# c5 = np.array([0.678565,  0.131021,  -0.136638, -0.158558 , 0.082139])*1e6

# Cp_mol = SX.zeros(N)
# for i in range(N):  # Loop over spatial points
#     #Cp_mol[i] = 0  # Initialize specific heat for the mixture
#     for j in range(ncomp):  # Loop over components
#         Cp_mol[i] += (
#             c1[j]
#             + c2[j] * T[i]
#             + c3[j] * T[i]**2
#             + c4[j] * T[i]**3
#             + c5[j] / T[i]**2
#         )/MW[j]*1000

omega = vertcat(wCH4, wCO, wCO2, wH2, wH2O)

# # Mixture specific heat at each spatial point
# Cpmix = SX.zeros(N)
# for i in range(0,N):
#     # Extract omega for the i-th spatial point
#     omega_i = omega[i::N]  # Get mass fractions for all components at spatial point i
#     Cpmix[i] = sum1(Cp_mol[i] * omega_i)  # Dot product of Cp and omega_i# Element-wise multiplication and summation

# Only constant value 
Cpmix = 2250 
# Dh_reaction = SX.zeros((3, N))
# for i in range(0, N): 
#     term1 = 0 
#     for j in range(0, n_comp):    
#         term1 += nu[:,j] * (
#             c1[j] * (T[i] - 298)
#             + c2[j] * (T[i]**2 - 298**2) / 2
#             + c3[j] * (T[i]**3 - 298**3) / 3
#             + c4[j] * (T[i]**4 - 298**4) / 4
#             - c5[j] * (1 / T[i] - 1 / 298)
#         )  # J/mol
#     Dh_reaction[:, i] = DHreact * 1000 + term1
# DH_reaction = Dh_reaction * 1000  # J/kmol

Dh_reaction = SX.zeros(3,N) + [225054923.824, -35038982.22,190015941.6]
######## Simplifications ############ 
Eta = SX.zeros(3,N) + 0.1
Tf = 850+273.15         # Constant value as first approximation
P = DM.zeros(N) + np.ones(N)*Pin_R1                  # Constant pressure 

# Pi = SX.zeros(N*n_comp)
# yi = SX.zeros(N*ncomp)

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
u = (F3) * R * T / (Aint*Ppa)                                                   # Superficial Gas velocity if the tube was empy (Coke at al. 2007)
u = VolFlow_R1 / (Aint * Epsilon)                                               # Gas velocity in the tube [m/s]
vz  = VolFlow_R1 / (Aint * Epsilon)                                               # Gas velocity in the tube [m/s]
rj,kr = Kinetics(T,R, Pi, RhoC, Epsilon,N)

#rj= SX.zeros(3,N) + [875248.66, 0.000104 ,10147109677.00]

# Constant U based on ss results 
U = 60
# U,lambda_gas,DynVis = HeatTransfer(T,Tc,n_comp, MW, Pc, yi, Cpmix, RhoGas,dTube, Dp, Epsilon, e_w, u, dTube_out, lambda_s)
#Deff = Diffusivity(R,T,P,yi, n_comp, MW, MWmix, e_s, tau,N)
#Eta = EffectivenessF(p_h,c_h,n_h,s_h,Dp,lambda_gas,Deff,kr)

# rj should have size N
const = Aint/(m_gas*3600)
# Define nu_expanded using CasADi's repmat
# Use element-wise multiplication and summation
rhs1 = -vz*dCH4dz + (vz.T * (const * MW[0] * sum1(repmat(nu[:, 0], 1, N) * (Eta * rj)))).T
rhs2 = -vz*dCOdz  + (vz.T*(const*MW[1] * sum1(repmat(nu[:, 1], 1, N) * (Eta * rj)))).T
rhs3 = -vz*dCO2dz + (vz.T*(const*MW[2] * sum1(repmat(nu[:, 2], 1, N) * (Eta * rj)))).T
rhs4 = -vz*dH2dz  + (vz.T*(const*MW[3] * sum1(repmat(nu[:, 3], 1, N) * (Eta * rj)))).T
rhs5 = -vz*dH2Odz + (vz.T*(const*MW[4] * sum1(repmat(nu[:, 4], 1, N) * (Eta * rj)))).T
rhs6 = (np.pi*dTube/(m_gas*Cpmix))*U*(Tf - T) -const/Cpmix* (sum1(Dh_reaction* (Eta*rj))).T
# DynVis should have size N 
#rhs7 = ( (-150 * (((1-Epsilon)**2)/(Epsilon**3)) * DynVis * u/ (Dp**2) - (1.75* ((1-Epsilon)/(Epsilon**3)) * m_gas*u/(Dp*Aint))  ) ) / 1e5

ode = vertcat(dCH4dt - rhs1, dCOdt - rhs2, dCO2dt - rhs3, dH2dt - rhs4, dH2Odt - rhs5, dTdt - rhs6)
alg = vertcat()
#alg = vertcat( dPdz-rhs7 )

# Define the DAE system correctly
dae = {
    'x': vertcat(wCH4, wCO, wCO2, wH2, wH2O, T),  # Differential states
    'z': vertcat(),                              # Algebraic variables (if any)
    'p': (),                                      # Parameters (if any)
    'ode': ode,                                  # Ordinary differential equations
    'alg': alg                                   # Algebraic equations (if any)
}

X0 = vertcat(w0_CH4, w0_CO, w0_CO2, w0_H2, w0_H2O, T0)

T_end = 1.0  # Simulation time in hours 
time_points = np.linspace(0, T_end, 5)  # Intermediate time points

collocation_opts = {
        'tf': 1/N,
        'number_of_finite_elements': 1,
        'collocation_scheme':'radau',
        # 'rootfinder': 'fast_newton',
        'interpolation_order': 4,
        'rootfinder_options':
            {'line_search': False, 'abstolStep': 1e-4, 'max_iter': 20, 'print_iteration': False} #, 'abstol': TOL

        # 'jit': True #   #error Code generation not supported for Collocation
    }

# CASADOS
t0 = 0 
_, f_casados, l_casados = create_awe_casados_integrator(dae, time_points, collocation_opts=collocation_opts, record_time=True, with_sensitivities=True, use_cython=True)
print(f"time to create casados integrator {time.time() - t0} s")
# x_sim_casados, l_sim_casados, timings_casados = run_simulation(f_casados, l_casados, ca.vertcat(X0, 0.0).full().squeeze(), controls, N_sim)
# jacs_casados, timings_jac_casados = run_jacobian_test(f_casados, x_sim_casadi, controls)

# Integrate over time
# n_steps = 5  # Number of time steps

# # Convert results to numpy array for post-processing
# #results = np.array(results)

# time = np.linspace(0, 1.0, n_steps)  # Adjust time vector
# for i in range(N):
#     plt.plot(time, results[:, i], label=f"CH4 at z{i}")
# plt.legend()
# plt.xlabel("Time [s]")
# plt.ylabel("CH4 Concentration")
# plt.show()
# # Simulate the system
# time_steps = np.linspace(0, 10, 100)  # Define time grid for simulation
# states_over_time = []

# for t in time_steps:
#     integrator.set("t", t)
#     integrator.set("x", x0)
#     integrator.solve()
#     x0 = integrator.get("x")  # Update the state for the next time step
#     states_over_time.append(x0.full())

# # Convert results to NumPy array for analysis
# states_over_time = np.array(states_over_time)

# # Plot the results
# plt.figure(figsize=(10, 6))
# for i in range(N):  # Plot for each spatial point
#     plt.plot(time_steps, states_over_time[:, i], label=f"CH4 - Spatial Point {i}")
# plt.xlabel("Time [s]")
# plt.ylabel("CH4 Concentration")
# plt.legend()
# plt.title("CH4 Concentration Over Time")
# plt.show()