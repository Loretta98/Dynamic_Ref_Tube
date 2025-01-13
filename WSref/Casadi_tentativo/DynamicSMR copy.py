# Dynamic model with finite differences (implicit) expression of the spatial derivatives 
# The method of lines is applied to discretize over space and integrate over time 
# The following simplifications are applied: Tw,Tf = const., eta = 0.1, no axial dispersion 

from Properties import * 
import numpy as np 
from casadi import * 
from Input import * 
import matplotlib.pyplot as plt 

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

dCH4dz = SX.zeros(N)
dCOdz = SX.zeros(N)
dCO2dz = SX.zeros(N)
dH2dz = SX.zeros(N) 
dH2Odz = SX.zeros(N)
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
print(P)
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
rj,kr = Kinetics(T,R, Pi, RhoC, Epsilon,N)


# Constant U based on ss results 
U = 60
# U,lambda_gas,DynVis = HeatTransfer(T,Tc,n_comp, MW, Pc, yi, Cpmix, RhoGas,dTube, Dp, Epsilon, e_w, u, dTube_out, lambda_s)
#Deff = Diffusivity(R,T,P,yi, n_comp, MW, MWmix, e_s, tau,N)
#Eta = EffectivenessF(p_h,c_h,n_h,s_h,Dp,lambda_gas,Deff,kr)

# rj should have size N
const = Aint/(m_gas*3600)
# Define nu_expanded using CasADi's repmat
# Use element-wise multiplication and summation
rhs1 = -vz*dCH4dz + (vz * (const * MW[0] * sum1(repmat(nu[:, 0], 1, N) * (Eta * rj)))).T
rhs2 = -vz*dCOdz  + (vz*(const*MW[1] * sum1(repmat(nu[:, 1], 1, N) * (Eta * rj)))).T
rhs3 = -vz*dCO2dz + (vz*(const*MW[2] * sum1(repmat(nu[:, 2], 1, N) * (Eta * rj)))).T
rhs4 = -vz*dH2dz  + (vz*(const*MW[3] * sum1(repmat(nu[:, 3], 1, N) * (Eta * rj)))).T
rhs5 = -vz*dH2Odz + (vz*(const*MW[4] * sum1(repmat(nu[:, 0], 1, N) * (Eta * rj)))).T
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

# Set the initial condition for the integrator
# Initial values for mass fractions (wCH4, wCO, etc.) and temperature (T)
w0_CH4 = DM([w0[0]] * N)  # Replace `wCH4_initial` with the actual initial value
w0_CO = DM([w0[1]] * N)    # Replace `wCO_initial` with the actual initial value
w0_CO2 = DM([w0[2]] * N)  # Replace `wCO2_initial` with the actual initial value
w0_H2 = DM([w0[3]] * N)    # Replace `wH2_initial` with the actual initial value
w0_H2O = DM([w0[4]] * N)  # Replace `wH2O_initial` with the actual initial value
T0 = DM([Tin_R1] * N)            # Initial temperature for all spatial points

# Stack the initial states together
T0 = DM([973.15,844.49873485,883.16560916,921.2683747,956.77900731,989.61761612,1019.58960853,1046.60074233,1069.54351148,1087.56602408])

w0_CH4 = DM([9.94454031e-02, 6.36089847e-02, 5.31489777e-02, 4.16291511e-02, 3.06863485e-02, 2.13230089e-02, 1.40311139e-02, 8.98783946e-03, 5.88853533e-03, 4.17853376e-03])
w0_CO = DM([1.36879324e-05, 1.52244465e-02, 2.58357529e-02, 4.08327141e-02, 5.82066178e-02, 7.57771249e-02, 9.18217516e-02, 1.04944288e-01, 1.14650592e-01, 1.21169960e-01])
w0_CO2 = DM([2.23244897e-01, 2.97672111e-01, 3.09699173e-01, 3.17743283e-01, 3.20469464e-01, 3.18553061e-01, 3.13350600e-01, 3.06569711e-01, 2.99822692e-01, 2.94271145e-01])
w0_H2 = DM([9.85179281e-07, 1.69227151e-02, 2.14176706e-02, 2.61297887e-02, 3.03807332e-02, 3.38234632e-02, 3.63346127e-02, 3.79255990e-02, 3.87851485e-02, 3.91756132e-02])
w0_H2O = DM([6.77295026e-01, 6.06579037e-01, 5.89907691e-01, 5.73676413e-01, 5.60270087e-01, 5.50538149e-01, 5.44477880e-01, 5.41589265e-01, 5.40870150e-01, 5.41222065e-01])
X0 = vertcat(w0_CH4, w0_CO, w0_CO2, w0_H2, w0_H2O, T0)

T_end = 1.0  # Simulation time in hours 
time_points = np.linspace(0, T_end, 5)  # Intermediate time points

# Casados integrators 

# Set up Casados integrator options
options = {
    'grid': time_points.tolist(),  # Define the time grid for integration
    'abstol': 1e-6,               # Absolute tolerance (adjust if needed)
    'reltol': 1e-6,               # Relative tolerance (adjust if needed)
    'max_num_steps': 5000         # Maximum number of steps
}

options = {
    'grid': time_points.tolist(),
    'abstol': 1e-2,  # Absolute tolerance
    'reltol': 1e-2,  # Relative tolerance
    'max_num_steps': 10000  # Increase max steps if needed
}

integrator = integrator('integrator', 'idas', dae, options)

# # Assuming the integrator produces results over a time grid and outputs state variables
results = integrator(x0 = X0)  # Execute the integrator to obtain results

# Extract solutions for species and temperature profiles
solution = results['xf'].full()  # Assuming 'xf' contains the final states after integration

species_profiles = []
temperature_profiles = []
time_steps = np.linspace(0, T_end, len(time_points))  # Define time steps

for t_idx, t in enumerate(time_steps):
    # Extract species and temperature at each time step
    wCH4_values = solution[t_idx, :N]  # CH4 mass fractions
    wCO_values = solution[t_idx, N:2*N]  # CO mass fractions
    wCO2_values = solution[t_idx, 2*N:3*N]  # CO2 mass fractions
    wH2_values = solution[t_idx, 3*N:4*N]  # H2 mass fractions
    wH2O_values = solution[t_idx, 4*N:5*N]  # H2O mass fractions
    temperature_values = solution[t_idx, 5*N:6*N]  # Temperature profile

    species_profiles.append([
        wCH4_values,
        wCO_values,
        wCO2_values,
        wH2_values,
        wH2O_values
    ])
    temperature_profiles.append(temperature_values)

species_profiles = np.array(species_profiles)  # Shape: [time_steps, n_species, reactor_length]
temperature_profiles = np.array(temperature_profiles)  # Shape: [time_steps, reactor_length]

# Compute time-averaged temperature
average_temperature = np.mean(temperature_profiles, axis=0)
reactor_length = np.linspace(0, L, N)  # Define reactor length points
time_steps = []  # To store time steps
# Plot species evolution
for species_idx, species_name in enumerate(['CH4', 'CO', 'CO2', 'H2', 'H2O']):
    plt.figure()
    for t_idx, t in enumerate(time_steps):
        plt.plot(reactor_length, species_profiles[t_idx, species_idx], label=f'Time {t:.2f}s')
    plt.title(f'{species_name} Evolution Along Reactor Length')
    plt.xlabel('Reactor Length (m)')
    plt.ylabel(f'{species_name} Mass Fraction')
    plt.legend()
    plt.grid()
    plt.show()

# Plot temperature evolution as a heatmap
plt.figure()
plt.imshow(
    temperature_profiles, 
    aspect='auto', 
    extent=[0, L, time_steps[-1], time_steps[0]], 
    cmap='hot'
)
plt.colorbar(label='Temperature (K)')
plt.title('Temperature Evolution')
plt.xlabel('Reactor Length (m)')
plt.ylabel('Time (s)')
plt.show()

# Plot time-averaged temperature profile
plt.figure()
plt.plot(reactor_length, average_temperature, label='Time-Averaged Temperature')
plt.title('Time-Averaged Temperature Profile Along Reactor Length')
plt.xlabel('Reactor Length (m)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.grid()
plt.show()
