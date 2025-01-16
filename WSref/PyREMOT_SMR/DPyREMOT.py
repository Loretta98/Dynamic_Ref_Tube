# Dynamic model for catalytic reformer reactor (central differences approximation)
from Input import * 
import numpy as np 

### IMPORT PACKAGES/MODULES ###
#------------------------------
from PyREMOT import rmtCom

### IMPORT PACKAGES/MODULES ###
#------------------------------
import math
from PyREMOT import rmtExe

### OPERATING CONDITIONS ###
#----------------------------
# pressure [Pa]
P = 15 * 1e5
# temperature [K]
T = 600+273.15
# process-type [-]
PrTy = "non-iso-thermal"
# operation-period [s]
opT = 0.5

### FEED PROPERTIES ###
#-----------------------
# flowrate @ P & T [m^3/s]
VoFlRa = VolFlow_R1/3600
# species-concentration [mol/m^3] 
SpCoi = Fin/(VoFlRa*3600)

### REACTOR SPEC ###
#-------------------
# reactor-length [m]
ReLe = 2
# reactor-inner-diameter [m]
ReInDi =  0.14142
# bed-void-fraction [-]
BeVoFr = 0.519
# catalyst bed density [kg/m^3]
CaBeDe = 2355.2
# particle-diameter [m]
PaDi = 0.015
# particle-density [kg/m^3]
CaDe = 1920
# particle-specific-heat-capacity  [J/kg.K]
CaSpHeCa = 960

### HEAT-EXCHANGER SPEC ###
#---------------------------
# overall-heat-transfer-coefficient [J/m^2.s.K]
U = 60
# medium-temperature [K]
Tm = 600+273.15

### SOLVER SETTING ###
#---------------------
# ode-solver [-]
ivp = "default"
# display-result [-]
diRe = "True"

### COMPONENT LIST ###
#----------------------
compList = ["H2","CO","CO2","H2","H2O"]

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

### REACTION RATE PARAMS ###
#----------------------------
varis0 = {
  "CaBeDe" : CaBeDe,
  "RT": lambda x: x['R_CONST']*x['T'],
  "K1": lambda x: math.exp(gamma[0,0]/(x['T']**3)+gamma[0,1]/(x['T']**2)+gamma[0,2]/x['T'] + gamma[0,3]),
  "K2": lambda x: math.exp(gamma[1,0]/(x['T']**3)+gamma[1,1]/(x['T']**2)+gamma[1,2]/x['T'] + gamma[1,3]),
  "K3": lambda x: x['K1']*x['K2'],
  "KH2": lambda x: 0.249*math.exp(3.4394e4/x['RT']),
  "KCO2": lambda x: 1.02e-7*math.exp(6.74e4/x['RT']),
  "KCO": lambda x: 7.99e-7*math.exp(5.81e4/x['RT']),
  "Ln_KP1": lambda x: 4213/x['T'] - 5.752 *     math.log(x['T']) - 1.707e-3*x['T'] + 2.682e-6 *     (math.pow(x['T'], 2)) - 7.232e-10*(math.pow(x['T'], 3)) + 17.6,
  "KP1": lambda x: math.exp(x['Ln_KP1']),
  "log_KP2": lambda x: 2167/x['T'] - 0.5194 *     math.log10(x['T']) + 1.037e-3*x['T'] - 2.331e-7 *     (math.pow(x['T'], 2)) - 1.2777,
  "KP2": lambda x: math.pow(10, x['log_KP2']),
      "Ln_KP3": lambda x:  4019/x['T'] + 3.707 *     math.log(x['T']) - 2.783e-3*x['T'] + 3.8e-7 *     (math.pow(x['T'], 2)) - 6.56e-4/(math.pow(x['T'], 3)) - 26.64,
  "KP3": lambda x:  math.exp(x['Ln_KP3']),
  "yi_CH4": lambda x:  x['MoFri'][0],
  "yi_CO": lambda x:  x['MoFri'][1],
  "yi_CO2": lambda x:  x['MoFri'][2],
  "yi_H2": lambda x:  x['MoFri'][3],
  "yi_H2O": lambda x:  x['MoFri'][4],
  "PH2": lambda x:  x['P']*(x['yi_H2'])*1e-5,
  "PCO2": lambda x:  x['P']*(x['yi_CO2'])*1e-5,
  "PH2O": lambda x:  x['P']*(x['yi_H2O'])*1e-5,
  "PCO": lambda x: x['P']*(x['yi_CO'])*1e-5,
  "PCH4": lambda x: x['P']*(x['yi_CH4'])*1e-5,
  "ra1": lambda x:  x['PCO2']*x['PH2'],
  "ra2": lambda x:  1 + (x['KCO2']*x['PCO2']) + (x['KCO']*x['PCO']) + math.sqrt(x['KH2']*x['PH2']),
  "ra3": lambda x: (1/x['KP1'])*((x['PH2O']*x['PCH3OH'])/(x['PCO2']*(math.pow(x['PH2'], 3)))),
  "ra4": lambda x:  x['PH2O'] - (1/x['KP2'])*((x['PCO2']*x['PH2'])/x['PCO']),
  "ra5": lambda x: (math.pow(x['PCH3OH'], 2)/x['PH2O'])-(x['PCH3OCH3']/x['KP3'])
}

### REACTION RATE EXPRESSIONS ###
#--------------------------------
rates0 = {
  "r1": lambda x: 1000*x['K1']*(x['ra1']/(math.pow(x['ra2'], 3)))*(1-x['ra3'])*x['CaBeDe'],
  "r2": lambda x: 1000*x['K2']*(1/x['ra2'])*x['ra4']*x['CaBeDe'],
  "r3": lambda x: 1000*x['K3']*x['ra5']*x['CaBeDe']
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
