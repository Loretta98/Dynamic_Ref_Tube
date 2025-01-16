import numpy as np 
from casadi import*

# INPUT DATA FIRST REACTOR
n_comp = 5; 
nu = np.array([ [-1, 1, 0, 3, -1],
                [0, -1, 1, 1, -1], 
                [-1, 0, 1, 4, -2]])  # SMR, WGS, reverse methanation

# Components  [CH4, CO, CO2, H2, H2O] O2, N2]
MW = np.array([16.04, 28.01, 44.01, 2.016, 18.01528 ]) #, 32.00, 28.01])                    # Molecular Molar weight       [kg/kmol]
Tc = np.array([-82.6, -140.3, 31.2, -240, 374]) + 273.15            # Critical Temperatures [K]
Pc = np.array([46.5, 35, 73.8, 13, 220.5])                          # Critical Pressures [bar]

# Components  [CH4, CO, CO2, H2, H2O] O2, N2]
MW_f = np.array([16.04, 28.01, 44.01, 2.016, 18.01528, 32.00, 28.01])                    # Molecular Molar weight       [kg/kmol]
Tc_f = np.array([-82.6, -140.3, 31.2, -240, 374,-118.6,-147]) + 273.15            # Critical Temperatures [K]
Pc_f = np.array([46.5, 35, 73.8, 13, 220.5,50.5,34])                          # Critical Pressures [bar]

# Data from FAT experimental setup 
Nt =   4                                                                                   # Number of tubes
dTube = 0.14142                                                                              # Tube diameter [m]
dTube_out = dTube+0.06                                                                          # Tube outlet diameter [m]
Length = 2                                                                                 # Length of the reactor [m]
s_w = 0.03

# Catalyst particle data
Epsilon = 0.519                                                                             # Void Fraction 
RhoC = 2355.2                                                                               # Catalyst density [kg/m3]
Dp = 0.015                                                                                 # Catalyst particle diameter [m]
p_h = 15                                                                                 # Pellet height [mm]
c_h = 5                                                                                  # central hole diameter [mm]
n_h = 0                                                                                     # number of side holes 
s_h = 0                                                                                     # side holes diameter [m]

tau = 3.54                                                                                  # Tortuosity 
e_s = 0.25                                                                                  # porosity of the catalyst particle [m3void/ m3cat] --> Tacchino
e_w = 0.85                                                                                  # emissivity of tube from Quirino
e_f = 0.3758                                                                                # emissitivty of the furnace from Quirino
lambda_s = 0.3489                                                                           # thermal conductivity of the solid [W/m/K]
sigma = 5.67e-8 # Stefan Boltzmann constan [W/m2/K4] 
k_w = 28.5 # tube thermal conductivity [W/mK]
p = dTube_out*(1+1)                                     # Tubes pitch [m]
D_h =  (dTube+s_w*2)*(4/np.pi*(p/(dTube/2+s_w)-1))    # Hydraulic Diameter [m]
Eta_list = []
kr_list = []
Deff_list = []
Tw_list = []                                                                         # input molar flowrate (kmol/s)

# Furnace data 
f_biogas = 30                           # Nm3/h Biogas available as fuel 
f_biogas = 30/22.41                     # kmol/h
excess = 0
f_air = ((f_biogas*0.6)*2)/0.21*(1+excess)  # Ratio of air for stochiometric combustion with a 5% excess of air
f_furnace = f_biogas + f_air            # kmol/h  
# [CH4, CO, CO2, H2, H2O, O2,N2]
x_f = np.array([0.093, 0.062, 0, 0, 0, 0.178, 0.667])   # composition at the inlet 
x_f = np.array([0, 0.06, 0.11, 0, 0.13, 0, 0.7])        # composition of the exhaust gas
p = dTube_out*(1+1)                                     # Tubes pitch [m]
D_h =  (dTube+s_w*2)*(4/np.pi*(p/(dTube/2+s_w)-1))    # Hydraulic Diameter [m]
A_f = 2*2- dTube_out**2/4*np.pi*Nt  # tranversal area of the furnace [m2] L = 2, W = 2 metri come guess
m_furnace = np.sum(np.multiply(f_furnace,np.multiply(x_f,MW_f))) # kg/h

# Components  [CH4, CO, CO2, H2, H2O]

#### INLET FROM REAL DATA !!!! ####
Twin = 600+273.15                                                                         # Tube wall temperature [K]
Tf = 850+273.15
Tin_R1 =  700+273.15                                                                            # Inlet Temperature [K]
Pin_R1 =  15                                                                              # Inlet Pressure [Bar]

# #x_in_R1 = np.array([0.22155701, 0.00, 0.01242592, 0.02248117, 0.74353591 ])                              # Inlet molar composition
# Fin = np.array([0.5439,0.0001,0.3461,0.0001,2.7039])    #kmol/h
# f_IN = np.sum(Fin)/Nt                                   # inlet molar flow per tube [kmol/h]
# x_in = np.zeros(n_comp)
# for i in range(0,n_comp):
#     x_in[i] = Fin[i]/np.sum(Fin)                     # inlet molar composition

# if measured data is: 0.5 CO2 e 0.5 CH4 in M1, M1 = 24 kg/h and M2 = 47.8 kg/h  all water
# Mtot = 71.8 kg/h 
# [CH4, CO, CO2, H2, H2O] O2, N2]
M1 = 23.85; M2 = 47.855              #kg/h

# Nominal conditions from DR 
M1 = 21.84
M2 = 45.84
x_M1_CH4 = 0.55
x_M1_CO2 = 1- x_M1_CH4
# Calculate molar flow rates (kmol/h) for M1 and M2
F1 = M1 / (x_M1_CH4 * MW[0] + x_M1_CO2 * MW[2])  # CH4 and CO2 in M1
F2 = M2 / MW[4]  # All H2O in M2

# Total molar flow rate (kmol/h)
F3 = F1 + F2
# Mole fractions at the inlet
x_in_CH4 = (F1 * x_M1_CH4) / F3
x_in_CO2 = (F1 * x_M1_CO2) / F3
x_in_H2O = F2 / F3

# Mole fraction vector for all species: [CH4, CO, CO2, H2, H2O}
x_in = np.zeros(5)
x_in[0] = x_in_CH4  # CH4
x_in[1] = 0.001
x_in[2] = x_in_CO2  # CO2
x_in[3] = 0.001
x_in[4] = x_in_H2O  # H2O

f_IN = F3/Nt # molar flowrate per tube kmol/h
Fin = F3*x_in                                            # molar inlet stream per component kmol/h
f_IN_i = x_in*f_IN                                       # inlet flowrate per component per tube
MWmix = np.sum(x_in*MW)
w0 = x_in*MW / MWmix

m_gas = (M1+M2)/3600/Nt  # mass flow kg/s per tube
SC = x_in[4] / x_in[0]        # the steam to carbon was calculated upon the total amount of carbon, not only methane
#print('Steam to Carbon ratio=', SC)

R = 8.314                                                                               # [J/molK]
Aint = np.pi*dTube**2/4                                                              # Tube section [m2]
# Perry's data 
        # CH4,          CO,             CO2,            H2,              H2O          
dH_formation_i = np.array([-74.52, -110.53, -393.51, 0, -241.814])                                  # Enthalpy of formation [kJ/mol]       
DHreact = np.sum(np.multiply(nu,dH_formation_i),axis=1).transpose()                                 # Enthalpy of reaction              [kJ/mol]

Mi_R1 = m_gas * w0 * 3600                                           # Mass flowrate per component per tube [kg/h]
RhoGas = (Pin_R1*1e5*MWmix) / (R*Tin_R1)  / 1000                                        # Gas mass density [kg/m3]
VolFlow_R1 = m_gas*3600 / RhoGas    # m3/h per tube 
vz = VolFlow_R1                    /Aint ### oppure potrebbe essere definito come u 

w0_CH4 = DM([9.94454031e-02, 6.36089847e-02, 5.31489777e-02, 4.16291511e-02, 3.06863485e-02, 2.13230089e-02, 1.40311139e-02, 8.98783946e-03, 5.88853533e-03, 4.17853376e-03])
w0_CO = DM([1.36879324e-05, 1.52244465e-02, 2.58357529e-02, 4.08327141e-02, 5.82066178e-02, 7.57771249e-02, 9.18217516e-02, 1.04944288e-01, 1.14650592e-01, 1.21169960e-01])
w0_CO2 = DM([2.23244897e-01, 2.97672111e-01, 3.09699173e-01, 3.17743283e-01, 3.20469464e-01, 3.18553061e-01, 3.13350600e-01, 3.06569711e-01, 2.99822692e-01, 2.94271145e-01])
w0_H2 = DM([9.85179281e-07, 1.69227151e-02, 2.14176706e-02, 2.61297887e-02, 3.03807332e-02, 3.38234632e-02, 3.63346127e-02, 3.79255990e-02, 3.87851485e-02, 3.91756132e-02])
w0_H2O = DM([6.77295026e-01, 6.06579037e-01, 5.89907691e-01, 5.73676413e-01, 5.60270087e-01, 5.50538149e-01, 5.44477880e-01, 5.41589265e-01, 5.40870150e-01, 5.41222065e-01])
# Set the initial condition for the integrator
# # Initial values for mass fractions (wCH4, wCO, etc.) and temperature (T)
# w0_CH4 = DM([w0[0]] * N)  # Replace `wCH4_initial` with the actual initial value
# w0_CO = DM([w0[1]] * N)    # Replace `wCO_initial` with the actual initial value
# w0_CO2 = DM([w0[2]] * N)  # Replace `wCO2_initial` with the actual initial value
# w0_H2 = DM([w0[3]] * N)    # Replace `wH2_initial` with the actual initial value
# w0_H2O = DM([w0[4]] * N)  # Replace `wH2O_initial` with the actual initial value
# T0 = DM([Tin_R1] * N)            # Initial temperature for all spatial points

# Stack the initial states together
T0 = DM([973.15,844.49873485,883.16560916,921.2683747,956.77900731,989.61761612,1019.58960853,1046.60074233,1069.54351148,1087.56602408])