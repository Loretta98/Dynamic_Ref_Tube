# Solving One Layer Burgers Equations.
# PDE: u_t = eps*u_xx - u*ux, with initial and boundary conditions
# defined from the exact solution.
# ------------------------------------------------------------------
# This example is based off a FORTRAN analogue for the original 
# BACOLI. The original can be found at:
#   http://cs.stmarys.ca/~muir/BACOLI95-3_Source/3-Problems/burg1.f
# ------------------------------------------------------------------

############## Dynamic Single PDE from Matlab ##################

import bacoli_py
import numpy 
from numpy import tanh
from Input import* 

# Initialize the Solver object.
solver = bacoli_py.Solver()

# Specify the number of PDE's in this system.
npde = 5

# Initialize problem-dependent parameters.
P = 15 #bar 
T = 800+273.15 #K # constant temperature 
Eta = 0.1                                               # effectiveness factor (Latham et al., Kumar et al.) 
# Mixture massive Specific Heat calculation (Shomate Equation)                                                        # Enthalpy of the reaction at the gas temperature [J/mol]
        # CH4,          CO,             CO2,            H2,              H2O    
c1 = np.array([-0.7030298, 25.56759,  24.99735, 33.066178 , 30.09 ])
c2 = np.array([108.4773,   6.096130,  55.18696, -11.363417, 6.832514])/1000
c3 = np.array([-42.52157,  4.054656, -33.69137, 11.432816 ,  6.793435])/1e6
c4 = np.array([5.862788, -2.671301,   7.948387, -2.772874 ,  -2.534480])/1e9
c5 = np.array([0.678565,  0.131021,  -0.136638, -0.158558 , 0.082139])*1e6
DH_reaction = DHreact*1000 + np.sum(nu*(c1*(T-298) + c2*(T**2-298**2)/2 + c3*(T**3-298**3)/3 + c4*(T**4-298**4)/4 - c5*(1/T-1/298)),1) #J/mol
DH_reaction = DH_reaction*1000 #J/kmol 
const = Aint/(m_gas*3600)

#Equilibrium constants fitted from Nielsen in [bar]
gamma = np.array([[-757.323e5, 997.315e3, -28.893e3, 31.29], [-646.231e5, 563.463e3, 3305.75,-3.466]]) #[bar^2], [-]
Keq1 = np.exp(gamma[0,0]/(T**3)+gamma[0,1]/(T**2)+gamma[0,2]/T + gamma[0,3])
Keq2 = np.exp(gamma[1,0]/(T**3)+gamma[1,1]/(T**2)+gamma[1,2]/T + gamma[1,3])
Keq3 = Keq1*Keq2
# Arrhenius     I,       II,        III
Tr_a = 648                                                                          # K 
Tr_b = 823                                                                          # K 
k0 = np.array([1.842e-4, 7.558,     2.193e-5])                                      # pre exponential factor @648 K kmol/bar/kgcat/h
E_a = np.array([240.1,   67.13,     243.9])                                         # activation energy [kJ/mol]
kr = k0*np.exp(-(E_a*1000)/R*(1/T-1/Tr_a))
#kr_list.append(kr)
# Van't Hoff    CO,     H2,         CH4,    H2O
K0_a = np.array([40.91,   0.0296])                                      # pre exponential factor @648 K [1/bar]
DH0_a = np.array([-70.65, -82.90])                                      # adsorption enthalpy [kJ/mol]
K0_b = np.array([0.1791, 0.4152])                                       # pre exponential factor @823 K [1/bar, -]

DH0_b = np.array([-38.28,  88.68])                                # adsorption enthalpy [kJ/mol]

Kr_a = K0_a*np.exp(-(DH0_a*1000)/R*(1/T-1/Tr_a))
Kr_b = K0_b*np.exp(-(DH0_b*1000)/R*(1/T-1/Tr_b))
#   CO, H2, CH4, H2O
Kr = np.concatenate((Kr_a,Kr_b)) # [1/bar] unless last one [-]
nt = 10
nx = 10

# Reactor definition 
def f(t, x, u, ux, uxx, fval):
    Pi = P*u                                               # Partial Pressure 
    DEN = 1 + Kr[0]*Pi[1] + Kr[1]*Pi[3] + Kr[2]*Pi[0] + Kr[3]*Pi[4]/Pi[3]
    # For example, use:
    nu = np.array([ [-1, 1, 0, 3, -1],
                [0, -1, 1, 1, -1], 
                [-1, 0, 1, 4, -2]])
     # Repeat `nu[:, 0]` along the second axis.
    rj = np.array([ (kr[0]/Pi[3]**(2.5)) * (Pi[0]*Pi[4]-(Pi[3]**3)*Pi[1]/Keq1) / DEN**2 , (kr[1]/Pi[3]) * (Pi[1]*Pi[4]-Pi[3]*Pi[2]/Keq2) / DEN**2 , (kr[2]/Pi[3]**(3.5)) * (Pi[0]*(Pi[4]**2)-(Pi[3]**4)*Pi[2]/Keq3) / DEN**2 ]) * RhoC * (1-Epsilon)  # kmol/m3/h
    # Add an axis to prepare for broadcasting and repeat 200 times along the last axis
    nu_expanded = np.repeat(nu[:, :, np.newaxis], rj.shape[1], axis=2)
    #nu_expanded = np.repeat(nu[:, :, np.newaxis], rj.shape, axis=2)  # Shape becomes (3, 5, 200)

    fval[0] = -vz*ux[0] + Aint*vz/Fin[0] * np.sum(np.multiply(nu_expanded[:, 0,:], np.multiply(Eta, rj)))
    fval[1] = -vz*ux[1] + Aint*vz/Fin[1] * np.sum(np.multiply(nu_expanded[:, 1,:], np.multiply(Eta, rj)))
    fval[2] = -vz*ux[2] + Aint*vz/Fin[2] * np.sum(np.multiply(nu_expanded[:, 2,:], np.multiply(Eta, rj)))
    fval[3] = -vz*ux[3] + Aint*vz/Fin[3] * np.sum(np.multiply(nu_expanded[:, 3,:], np.multiply(Eta, rj)))
    fval[4] = -vz*ux[4] + Aint*vz/Fin[4] * np.sum(np.multiply(nu_expanded[:, 4,:], np.multiply(Eta, rj)))
    return fval

# Function defining the left spatial boundary condition.
def bndxa(t, u, ux, bval):
    bval[0] = ux[0]
    bval[1] = ux[1]
    bval[2] = ux[2]
    bval[3] = ux[3]
    bval[4] = ux[4]
    return bval

# Function defining the right spatial boundary condition.
def bndxb(t, u, ux, bval):
    bval[0] = ux[0]
    bval[1] = ux[1]
    bval[2] = ux[2]
    bval[3] = ux[3]
    bval[4] = ux[4]
    return bval

# Function defining the initial conditions.
def uinit(x, u):
    u[0] = x_in[0]
    u[1] = x_in[1]
    u[2] = x_in[2]
    u[3] = x_in[3]
    u[4] = x_in[4]
    return u 

# Pack all of these callbacks and the number of PDE's into a 
# ProblemDefinition object.
problem_definition = bacoli_py.ProblemDefinition(npde, f=f, 
                                            bndxa=bndxa, 
                                            bndxb=bndxb,
                                            uinit=uinit)

# Specify initial mesh, output_points and output_times.

# Set t0.
initial_time = 0.0

# Define the initial spatial mesh.
# initial_mesh = numpy.linspace(0, 1, 11)
initial_mesh = [0, 1]

# Choose output times and points.
tspan = numpy.linspace(0.0001, 2, nt)
xspan = numpy.linspace(0, 2, nx)

# Solve this problem.
evaluation = solver.solve(problem_definition, initial_time, initial_mesh,
                           tspan, xspan, atol=1e-6, rtol=1e-6)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Assuming evaluation.u contains the solution array of shape (num_components, num_x, num_t)
u = evaluation.u  # Shape: (5, len(xspan), len(tspan))

# Create coordinate arrays for space (xspan) and time (tspan)
T, X = np.meshgrid(tspan, xspan)

# Components to plot
components = ["CH4", "CO", "CO2", "H2", "H2O"]

# # Plotting style
# styling = {
#     'cmap': cm.viridis,
#     'linewidth': 0,
#     'antialiased': True
# }

# # Create individual plots for each component
# fig = plt.figure(figsize=(15, 10))
# for i, component in enumerate(components):
#     ax = fig.add_subplot(2, 3, i + 1, projection='3d')
#     Z = u[i, :, :]  # Extract the solution for the i-th component
#     ax.plot_surface(T[:-1], X[:-1], Z[:-1], **styling)
#     ax.set_title(component)
#     ax.set_xlabel('$t$ (time)')
#     ax.set_ylabel('$x$ (space)')
#     ax.set_zlabel(f'${component}(t, x)$')

# # Adjust layout
# plt.tight_layout()
# plt.show()

# Plot component distributions at fixed time intervals along the space axis
time_intervals = np.arange(0, max(tspan) + 0.1, 0.5)  # Every 0.5 in tspan
time_indices = [np.abs(tspan - t).argmin() for t in time_intervals]  # Closest indices in tspan

fig, axes = plt.subplots(len(components), 1, figsize=(10, 12), sharex=True)
for i, component in enumerate(components):
    ax = axes[i]
    for t_idx in time_indices:
        t_value = tspan[t_idx]
        ax.plot(xspan, u[i, :, t_idx], label=f"$t = {t_value:.1f}$")
    ax.set_title(f"{component} Distribution Along Space")
    ax.set_xlabel("$x$ (space)")
    ax.set_ylabel(f"${component}(x)$")
    ax.legend(loc="upper right")
    ax.grid(True)

# Adjust layout for 2D line plots
plt.tight_layout()
plt.show()