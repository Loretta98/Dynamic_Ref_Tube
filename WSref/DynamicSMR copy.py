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
N = 100                          # Number of spatial points
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
#rj,kr = Kinetics(T,R, Pi, RhoC, Epsilon,N)

rj  = SX.zeros(3,N) + [875248.66, 0.00104, 10147109677.00]
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
#rhs6 = (np.pi*dTube/(m_gas*Cpmix))*U*(Tf - T) -const/Cpmix* (sum1(Dh_reaction* (Eta*rj))).T
rhs6 = 0 
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
#T0 = DM([973.15,844.49873485,883.16560916,921.2683747,956.77900731,989.61761612,1019.58960853,1046.60074233,1069.54351148,1087.56602408])
T0 = DM.zeros(N) + 973.15
# N = 10 
# w0_CH4 = DM([9.94454031e-02, 6.36089847e-02, 5.31489777e-02, 4.16291511e-02, 3.06863485e-02, 2.13230089e-02, 1.40311139e-02, 8.98783946e-03, 5.88853533e-03, 4.17853376e-03])
# w0_CO = DM([1.36879324e-05, 1.52244465e-02, 2.58357529e-02, 4.08327141e-02, 5.82066178e-02, 7.57771249e-02, 9.18217516e-02, 1.04944288e-01, 1.14650592e-01, 1.21169960e-01])
# w0_CO2 = DM([2.23244897e-01, 2.97672111e-01, 3.09699173e-01, 3.17743283e-01, 3.20469464e-01, 3.18553061e-01, 3.13350600e-01, 3.06569711e-01, 2.99822692e-01, 2.94271145e-01])
# w0_H2 = DM([9.85179281e-07, 1.69227151e-02, 2.14176706e-02, 2.61297887e-02, 3.03807332e-02, 3.38234632e-02, 3.63346127e-02, 3.79255990e-02, 3.87851485e-02, 3.91756132e-02])
# w0_H2O = DM([6.77295026e-01, 6.06579037e-01, 5.89907691e-01, 5.73676413e-01, 5.60270087e-01, 5.50538149e-01, 5.44477880e-01, 5.41589265e-01, 5.40870150e-01, 5.41222065e-01])

# N = 100 
w0_CH4 = DM([
    0.0994454, 0.05648256, 0.05593869, 0.05592719, 0.0559266, 0.05592677,
    0.05592618, 0.05592696, 0.05592531, 0.05592764, 0.05592649, 0.05592688,
    0.0559266, 0.05592677, 0.05592602, 0.05592671, 0.05592628, 0.05592691,
    0.05592598, 0.05592693, 0.05592559, 0.05592774, 0.0559262, 0.05592703,
    0.05592641, 0.0559268, 0.05592614, 0.05592642, 0.05592614, 0.05592723,
    0.05592628, 0.05592661, 0.05592621, 0.05592753, 0.05592625, 0.05592706,
    0.05592637, 0.05592671, 0.05592644, 0.05592599, 0.05592675, 0.05592743,
    0.05592647, 0.05592583, 0.05592691, 0.05592703, 0.05592653, 0.05592692,
    0.05592654, 0.05592649, 0.05592676, 0.05592553, 0.0559273, 0.05592696,
    0.05592668, 0.05592679, 0.05592668, 0.05592647, 0.05592688, 0.05592662,
    0.05592676, 0.05592618, 0.05592695, 0.05592529, 0.05592766, 0.05592649,
    0.05592688, 0.0559266, 0.05592677, 0.05592603, 0.05592769, 0.05592628,
    0.0559269, 0.05592599, 0.05592691, 0.05592559, 0.05592776, 0.05592621,
    0.05592703, 0.05592641, 0.05592679, 0.05592616, 0.05592636, 0.0559262,
    0.05592682, 0.05592627, 0.0559266, 0.05592621, 0.05592755, 0.05592625,
    0.05592706, 0.05592638, 0.0559267, 0.05592646, 0.05592593, 0.05592681,
    0.0559274, 0.05592649, 0.0559261, 0.0559266
]) 

w0_CO = DM([
    1.36879324e-05, 2.23623756e-02, 2.27653732e-02, 2.27232418e-02,
    2.27195403e-02, 2.27196361e-02, 2.27187528e-02, 2.27185088e-02,
    2.27183606e-02, 2.27191252e-02, 2.27179602e-02, 2.27201926e-02,
    2.27190812e-02, 2.27194589e-02, 2.27189634e-02, 2.27193815e-02,
    2.27187549e-02, 2.27191494e-02, 2.27186936e-02, 2.27191001e-02,
    2.27183361e-02, 2.27201092e-02, 2.27185368e-02, 2.27192480e-02,
    2.27189447e-02, 2.27192827e-02, 2.27187922e-02, 2.27189691e-02,
    2.27187677e-02, 2.27197364e-02, 2.27188279e-02, 2.27191929e-02,
    2.27186650e-02, 2.27199521e-02, 2.27187515e-02, 2.27192225e-02,
    2.27189052e-02, 2.27193433e-02, 2.27189800e-02, 2.27185996e-02,
    2.27192756e-02, 2.27199262e-02, 2.27190248e-02, 2.27184269e-02,
    2.27191818e-02, 2.27193062e-02, 2.27190792e-02, 2.27192871e-02,
    2.27190647e-02, 2.27190261e-02, 2.27193183e-02, 2.27182600e-02,
    2.27196752e-02, 2.27192569e-02, 2.27194096e-02, 2.27193785e-02,
    2.27193925e-02, 2.27190414e-02, 2.27193999e-02, 2.27191467e-02,
    2.27193224e-02, 2.27186962e-02, 2.27192954e-02, 2.27183444e-02,
    2.27199111e-02, 2.27190821e-02, 2.27194652e-02, 2.27189692e-02,
    2.27193799e-02, 2.27187803e-02, 2.27200741e-02, 2.27188250e-02,
    2.27191552e-02, 2.27186683e-02, 2.27191609e-02, 2.27183115e-02,
    2.27201325e-02, 2.27185485e-02, 2.27192764e-02, 2.27189483e-02,
    2.27192690e-02, 2.27188296e-02, 2.27189342e-02, 2.27187378e-02,
    2.27194017e-02, 2.27188396e-02, 2.27192072e-02, 2.27187682e-02,
    2.27199231e-02, 2.27187870e-02, 2.27192451e-02, 2.27189163e-02,
    2.27193321e-02, 2.27189762e-02, 2.27185756e-02, 2.27193104e-02,
    2.27199318e-02, 2.27190377e-02, 2.27187520e-02, 2.27191098e-02
])

w0_CO2 = DM([0.2232449,  0.30601005, 0.3068691,  0.30696686, 0.30694888, 0.30696027,
 0.30692233, 0.30697231, 0.30686708, 0.30701505, 0.30694253, 0.30696705,
 0.30694951, 0.30696023, 0.30691242, 0.306957,   0.30692844, 0.30696906,
 0.30690966, 0.30697026, 0.30688482, 0.30702197, 0.30692378, 0.30697683,
 0.30693715, 0.3069617, 0.30691988, 0.30693812, 0.30691952, 0.30698976,
 0.30692838, 0.30695017, 0.30692443, 0.30700844, 0.30692672, 0.3069787,
 0.30693471, 0.30695619, 0.30693896, 0.30691095, 0.30695862, 0.30700196,
 0.30694101, 0.30689968, 0.30696847, 0.30697675, 0.30694464, 0.30696958,
 0.30694513, 0.30694237, 0.30695937, 0.30688128, 0.30699329, 0.30697223,
 0.30695432, 0.30696142,0.30695462, 0.3069405,  0.30696673, 0.3069502,
 0.30695961, 0.30692254, 0.30697185, 0.30686567, 0.30701667, 0.306942,
 0.30696745, 0.30694899, 0.30696032, 0.30691245, 0.30701846, 0.30692894,
 0.30696848, 0.30690984, 0.30696937, 0.30688455, 0.30702307, 0.30692412,
 0.30697674, 0.30693721, 0.30696131, 0.30692131, 0.30693477, 0.30692376,
 0.30696273, 0.30692793, 0.3069493, 0.30692425, 0.30700954, 0.30692709,
 0.30697845, 0.30693524, 0.30695542, 0.30694039, 0.30690713, 0.3069624,
 0.30699998, 0.30694232, 0.30691758, 0.30694908])

w0_H2 = DM([9.85179281e-07, 1.99917278e-02, 2.02361483e-02, 2.02449627e-02,
 2.02443616e-02, 2.02448190e-02, 2.02433047e-02, 2.02453008e-02,
 2.02411021e-02, 2.02470033e-02, 2.02441119e-02, 2.02450904e-02,
 2.02443904e-02, 2.02448182e-02, 2.02429097e-02, 2.02446922e-02,
 2.02435471e-02, 2.02451698e-02, 2.02427985e-02, 2.02452195e-02,
 2.02418089e-02, 2.02472798e-02, 2.02433638e-02, 2.02454804e-02,
 2.02438967e-02, 2.02448798e-02, 2.02432076e-02, 2.02439391e-02,
 2.02431931e-02, 2.02459969e-02, 2.02435454e-02, 2.02444177e-02,
 2.02433884e-02, 2.02467405e-02, 2.02434808e-02, 2.02455549e-02,
 2.02437992e-02, 2.02446569e-02, 2.02439695e-02, 2.02428544e-02,
 2.02447524e-02, 2.02464822e-02, 2.02440517e-02, 2.02423997e-02,
 2.02451450e-02, 2.02454767e-02, 2.02441955e-02, 2.02451910e-02,
 2.02442149e-02, 2.02441050e-02, 2.02447845e-02, 2.02416699e-02,
 2.02461353e-02, 2.02452968e-02, 2.02445824e-02, 2.02448659e-02,
 2.02445944e-02, 2.02440296e-02, 2.02450766e-02, 2.02444172e-02,
 2.02447929e-02, 2.02433132e-02, 2.02452827e-02, 2.02410459e-02,
 2.02470678e-02, 2.02440909e-02, 2.02451063e-02, 2.02443696e-02,
 2.02448219e-02, 2.02429109e-02, 2.02471398e-02, 2.02435684e-02,
 2.02451468e-02, 2.02428058e-02, 2.02451840e-02, 2.02417984e-02,
 2.02473236e-02, 2.02433774e-02, 2.02454771e-02, 2.02438991e-02,
 2.02448614e-02, 2.02432645e-02, 2.02438053e-02, 2.02433623e-02,
 2.02449152e-02, 2.02435279e-02, 2.02443829e-02, 2.02433812e-02,
 2.02467845e-02, 2.02434954e-02, 2.02455450e-02, 2.02438202e-02,
 2.02446261e-02, 2.02440264e-02, 2.02427019e-02, 2.02449030e-02,
 2.02464034e-02, 2.02441039e-02, 2.02431165e-02, 2.02443739e-02])

w0_H2O = DM([0.67729503, 0.59516192, 0.59419943, 0.5941465,  0.59415319, 0.59414872,
 0.59416359, 0.594144,   0.59418523, 0.59412727, 0.59415567, 0.59414606,
 0.59415293, 0.59414873, 0.59416747, 0.59414999, 0.5941612,  0.59414528,
 0.59416855, 0.5941448,  0.59417828, 0.59412455, 0.59416301, 0.59414223,
 0.59415778, 0.59414813, 0.59416454, 0.59415739, 0.59416468, 0.59413716,
 0.59416122, 0.59415267, 0.59416276, 0.59412985, 0.59416186, 0.5941415,
 0.59415873, 0.59415032, 0.59415707, 0.59416803, 0.59414937, 0.59413239,
 0.59415626, 0.59417246, 0.59414551, 0.59414226, 0.59415484,  0.59414507,
 0.59415465, 0.59415573, 0.59414907, 0.59417966, 0.59413579, 0.59414403,
 0.59415105, 0.59414827, 0.59415093, 0.59415647, 0.59414619, 0.59415267,
 0.59414898, 0.5941635,  0.59414418, 0.59418578, 0.59412663, 0.59415588,
 0.59414591, 0.59415314, 0.5941487,  0.59416745, 0.59412593, 0.594161,
 0.5941455,  0.59416848, 0.59414515, 0.59417838, 0.59412412, 0.59416288,
 0.59414227, 0.59415776, 0.59414831, 0.59416399, 0.5941587,  0.59416302,
 0.59414776, 0.59416139, 0.59415302, 0.59416283, 0.59412942, 0.59416172,
 0.5941416,  0.59415853, 0.59415062, 0.59415651, 0.59416953, 0.59414789,
 0.59413317, 0.59415575, 0.59416544, 0.5941531 ])

X0 = vertcat(w0_CH4, w0_CO, w0_CO2, w0_H2, w0_H2O, T0)

T_end = 1.0  # Simulation time in hours 
time_points = np.linspace(0, T_end, 10)  # Intermediate time points

# Casados integrators 

# Set up Casados integrator options
options = {
    'grid': time_points.tolist(),  # Define the time grid for integration
    'abstol': 1e-3,               # Absolute tolerance (adjust if needed)
    'reltol': 1e-3,               # Relative tolerance (adjust if needed)
    'max_num_steps': 5000         # Maximum number of steps
}
integrator = integrator('integrator', 'idas', dae, options)

# Multiple shooting: Iterate over each segment
species_profiles = []  # To store species profiles over time
temperature_profiles = []  # To store temperature profiles over time

for t_idx, t in enumerate(time_points[:-1]):
    result = integrator(x0=X0)  # Integrate from the current state
    X0 = result['xf']  # Update initial state for the next interval

    # Extract and store species and temperature profiles at this step
    species_profiles.append([
        X0[:N].full().flatten(),       # CH4
        X0[N:2*N].full().flatten(),   # CO
        X0[2*N:3*N].full().flatten(), # CO2
        X0[3*N:4*N].full().flatten(), # H2
        X0[4*N:5*N].full().flatten()  # H2O
    ])
    temperature_profiles.append(X0[5*N:].full().flatten())  # Temperature

# Convert profiles to numpy arrays for analysis
species_profiles = np.array(species_profiles)  # Shape: [time_steps, n_species, reactor_length]
temperature_profiles = np.array(temperature_profiles)  # Shape: [time_steps, reactor_length]

species_profiles = []
temperature_profiles = []
time_steps = np.linspace(0, T_end, len(time_points))  # Define time steps

# Compute time-averaged temperature profile
average_temperature = np.mean(temperature_profiles, axis=0)
reactor_length = np.linspace(0, 1.0, N)  # Reactor length (assuming normalized)

# Plot species evolution
for species_idx, species_name in enumerate(['CH4', 'CO', 'CO2', 'H2', 'H2O']):
    plt.figure()
    for t_idx, t in enumerate(time_points[:-1]):
        plt.plot(reactor_length, species_profiles[t_idx, species_idx], label=f'Time {t:.2f} h')
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
    extent=[0, 1.0, time_points[-1], time_points[0]], 
    cmap='hot'
)
plt.colorbar(label='Temperature (K)')
plt.title('Temperature Evolution')
plt.xlabel('Reactor Length (m)')
plt.ylabel('Time (h)')
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