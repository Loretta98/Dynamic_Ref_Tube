# Dynamic model for catalytic reformer reactor (central differences approximation)
from Input import * 
import numpy as np 

### IMPORT PACKAGES/MODULES ###
#------------------------------
from PyREMOT import rmtCom
print(rmtCom())
### IMPORT PACKAGES/MODULES ###
#------------------------------
import math
from PyREMOT import rmtExe

### OPERATING CONDITIONS ###
#----------------------------
# pressure [Pa]
P = 1500000
# temperature [K]
T = 873.15
# process-type [-]
PrTy = "non-iso-thermal"
#PrTy = "isothermal"
# operation-period [s]
opT = ((60)*60)*20

### FEED PROPERTIES ###
#-----------------------
# flowrate @ P & T [m^3/s]
VoFlRa = VolFlow_R1/3600
# species-concentration [mol/m^3] 
SpCoi = Fin/(VoFlRa*3600)

### REACTOR SPEC ###
#-------------------
# reactor-length [m]
ReLe = 2.0
# reactor-inner-diameter [m]
ReInDi =  0.14142
# bed-void-fraction [-]
BeVoFr = 0.519
# catalyst bed density [kg/m^3]
CaBeDe = 2355.2
# particle-diameter [m]
PaDi = 0.015
# particle-density [kg/m^3]
CaDe = 1000
# particle-specific-heat-capacity  [J/kg.K]
CaSpHeCa = 900

### HEAT-EXCHANGER SPEC ###
#---------------------------
# overall-heat-transfer-coefficient [J/m^2.s.K]
U = 60
# medium-temperature [K]
Tm = 873.15

### SOLVER SETTING ###
#---------------------
# ode-solver [-]
ivp = "default"
# display-result [-]
diRe = "True"

### COMPONENT LIST ###
#----------------------
compList = ["CH4","CO","CO2","H2","H2O"]

### REACTION LIST ###
#---------------------
reactionSet = {
  "R1":"H2O+CH4<=>3H2+CO",
  "R2":"CO+H2O<=>H2+CO2",
  "R3":"CH4+2H2O<=>CO2+2H2"
}
# Equilibrium constants fitted from Nielsen in [bar]
gamma = np.array([[-757.323e5, 997.315e3, -28.893e3, 31.29], [-646.231e5, 563.463e3, 3305.75,-3.466]]) #[bar^2], [-]
# Arrhenius     I,       II,        III
Tr_a = 648                                                                          
Tr_b = 823                                                                          
k0 = np.array([1.842e-4, 7.558,     2.193e-5])                                      # pre exponential factor @648 K kmol/bar/kgcat/h
E_a = np.array([240.1,   67.13,     243.9])                                         # activation energy [kJ/mol]

# Adsorption constants
K0_a = np.array([40.91, 0.0296])  # pre-exponential factors @648 K [1/bar]
DH0_a = np.array([-70.65, -82.90])  # adsorption enthalpy [kJ/mol]
K0_b = np.array([0.1791, 0.4152])                                 # pre exponential factor @823 K [1/bar, -]
DH0_b = np.array([-38.28,  88.68])                                # adsorption enthalpy [kJ/mol] 

### REACTION PARAMETERS ###
# Equilibrium constants fitted from Nielsen in [bar]
gamma = np.array([[-757.323e5, 997.315e3, -28.893e3, 31.29], 
                  [-646.231e5, 563.463e3, 3305.75, -3.466]])

# Arrhenius constants
Tr_a = 648  # K
k0 = np.array([1.842e-4, 7.558, 2.193e-5])  # pre-exponential factors [kmol/bar/kgcat/h]
E_a = np.array([240.1, 67.13, 243.9])  # activation energies [kJ/mol]

# Adsorption constants
K0_a = np.array([40.91, 0.0296])  # pre-exponential factors @648 K [1/bar]
DH0_a = np.array([-70.65, -82.90])  # adsorption enthalpy [kJ/mol]
K0_b = np.array([0.1791, 0.4152])  # pre-exponential factors @823 K [1/bar]
DH0_b = np.array([-38.28, 88.68])  # adsorption enthalpy [kJ/mol]
Eta = 0.1 
### VARIABLE DEFINITIONS ###
varis0 = {
    "Eta": Eta,
    "CaBeDe" : CaBeDe,
    "RT": lambda x: x['R_CONST'] * x['T'],
    "K1": lambda x: math.exp(gamma[0, 0] / (x['T']**3) + gamma[0, 1] / (x['T']**2) + gamma[0, 2] / x['T'] + gamma[0, 3]),
    "K2": lambda x: math.exp(gamma[1, 0] / (x['T']**3) + gamma[1, 1] / (x['T']**2) + gamma[1, 2] / x['T'] + gamma[1, 3]),
    "K3": lambda x: x['K1'] * x['K2'],

    "kr0": lambda x: k0[0] * math.exp(-E_a[0] * 1000 / x['RT']),
    "kr1": lambda x: k0[1] * math.exp(-E_a[1] * 1000 / x['RT']),
    "kr2": lambda x: k0[2] * math.exp(-E_a[2] * 1000 / x['RT']),

    "Kr0": lambda x: K0_a[0] * math.exp(-((DH0_a[0] * 1000) / x['R_CONST']) * (1 / x['T'] - 1 / Tr_a)),
    "Kr1": lambda x: K0_a[1] * math.exp(-((DH0_a[1] * 1000) / x['R_CONST']) * (1 / x['T'] - 1 / Tr_a)),
    "Kr2": lambda x: K0_b[0] * math.exp(-((DH0_b[0] * 1000) / x['R_CONST']) * (1 / x['T'] - 1 / Tr_b)),
    "Kr3": lambda x: K0_b[1] * math.exp(-((DH0_b[1] * 1000) / x['R_CONST']) * (1 / x['T'] - 1 / Tr_b)),

    "yi_CH4": lambda x: x['MoFri'][0],
    "yi_CO": lambda x: x['MoFri'][1],
    "yi_CO2": lambda x: x['MoFri'][2],
    "yi_H2": lambda x: x['MoFri'][3],
    "yi_H2O": lambda x: x['MoFri'][4],
    "PH2": lambda x: x['P'] * (x['yi_H2']) * 1e-5,
    "PCO2": lambda x: x['P'] * (x['yi_CO2']) * 1e-5,
    "PH2O": lambda x: x['P'] * (x['yi_H2O']) * 1e-5,
    "PCO": lambda x: x['P'] * (x['yi_CO']) * 1e-5,
    "PCH4": lambda x: x['P'] * (x['yi_CH4']) * 1e-5,

    "DEN": lambda x: 1 + x['Kr0'] * x['PCO'] + x['Kr1'] * x['PH2'] + x['Kr2'] * x['PCH4'] + x['Kr3'] * x['PH2O'] / x['PH2'],
    "r1_term": lambda x: (x['kr0'] / (x['PH2']**2.5)) * (x['PCH4'] * x['PH2O'] - (x['PH2']**3) * x['PCO'] / x['K1']),
    "r2_term": lambda x: (x['kr1'] / x['PH2']) * (x['PCO'] * x['PH2O'] - x['PH2'] * x['PCO2'] / x['K2']),
    "r3_term": lambda x: (x['kr2'] / (x['PH2']**3.5)) * (x['PCH4'] * (x['PH2O']**2) - (x['PH2']**4) * x['PCO2'] / x['K3'])
}


### REACTION RATE EXPRESSIONS ###
rates0 = {
    "r1": lambda x: (x['r1_term'] / (x['DEN']**2)) * x['CaBeDe']*x['Eta'],
    "r2": lambda x: (x['r2_term'] / (x['DEN']**2)) * x['CaBeDe']*x['Eta'],
    "r3": lambda x: (x['r3_term'] / (x['DEN']**2)) * x['CaBeDe']*x['Eta']
}


### MODEL INPUTS ###
#-------------------
modelInput = {
  "model": "N2",
  "operating-conditions": {
      "pressure": P,
      "temperature": T,
      "process-type": PrTy,
      "period": opT
  },
  "feed": {
      "volumetric-flowrate": VoFlRa,
      "concentration": SpCoi,
      "components": {
          "shell": compList,
      }
  },
  "reactions": reactionSet,
  "reaction-rates": {
    "VARS": varis0,
    "RATES": rates0
  },
  "external-heat": {
    "OvHeTrCo": U,
    "MeTe": Tm
  },
  "reactor": {
      "ReInDi": ReInDi,
      "ReLe": ReLe,
      "PaDi": PaDi,
      "BeVoFr": BeVoFr,
      "CaBeDe": CaBeDe,
      "CaDe": CaDe,
      "CaSpHeCa": CaSpHeCa
  },
  "solver-config": {
      "ivp": ivp,
      "display-result": diRe
  }
}



# the values are calculated over the single catalityc tube though 

### RUN MODELLING ###
#--------------------
res = rmtExe(modelInput)
