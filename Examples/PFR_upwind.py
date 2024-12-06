import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Reactor and simulation parameters
N = 10  # Number of spatial segments
L = 10.0  # Reactor length [m]
dz = L / N  # Spatial step size
v = 1.0  # Constant velocity [m/s]
num_species = 2  # Example: A + B -> Products
delta_H = -100  # Reaction enthalpy [kJ/mol]
Cp = 1.0  # Heat capacity [kJ/kg.K]
F_tot = 1.0  # Total flow rate [mol/s]
t_final = 5.0  # Final time [s]
dt = 0.1  # Time step for plotting

# Define symbolic variables
C = ca.MX.sym('C', num_species, N)  # Species concentrations at all segments
T = ca.MX.sym('T', N)  # Temperature at all segments
C_in = [1.0, 0.0]  # Inlet concentrations
T_in = 300  # Inlet temperature

# Reaction rate (example)
r = lambda C, T: -C[0]  # First-order w.r.t. species A

# Spatial discretization
dCdt = []
dTdt = []
for i in range(N):
    # Upwind scheme for spatial derivatives
    if i == 0:  # Inlet boundary condition
        dCdz = (C[:, i] - ca.MX(C_in)) / dz
        dTdz = (T[i] - T_in) / dz
    else:
        dCdz = (C[:, i] - C[:, i-1]) / dz
        dTdz = (T[i] - T[i-1]) / dz

    # Species balance
    reaction_rate = r(C[:, i], T[i])
    dCdt.append(-v * dCdz + reaction_rate)

    # Energy balance
    dTdt.append(-v * dTdz + delta_H / (F_tot * Cp) * reaction_rate)

# Convert to CasADi structure
dCdt = ca.vertcat(*dCdt)  # Combine all equations
dTdt = ca.vertcat(*dTdt)
dXdt = ca.vertcat(dCdt, dTdt)  # Full system dynamics

# Create ODE function
X = ca.vertcat(ca.reshape(C, -1, 1), T)  # Flatten state
ode = {'x': X, 'ode': dXdt}

# Define time integrator
opts = {'tf': dt}  # Time step
integrator = ca.integrator('integrator', 'cvodes', ode, opts)

# Initial conditions
C0 = ca.DM.zeros((num_species, N))  # Initial species concentrations
T0 = ca.DM.ones((N, 1)) * T_in  # Initial temperature
X0 = ca.vertcat(ca.reshape(C0, -1, 1), T0)

# Time-stepping loop
t = 0.0
X_t = X0

# Data storage for plotting
time_points = []
outlet_temps = []
spatial_profiles_t = {}

while t <= t_final:
    # Store data
    time_points.append(t)
    outlet_temps.append(float(X_t[-1]))  # Outlet temperature

    # Store spatial profiles for t = 0 and t = 5
    if np.isclose(t, 0) or np.isclose(t, 5):
        spatial_profiles_t[t] = {
            'T': X_t[-N:],  # Last N values are temperatures
            'C': ca.reshape(X_t[:num_species * N], (num_species, N)),  # First part is concentration
        }

    # Integrate to the next time step
    result = integrator(x0=X_t)
    X_t = result['xf']
    t += dt

# Plotting results
plt.figure(figsize=(10, 6))

# Time evolution of outlet temperature
plt.subplot(2, 1, 1)
plt.plot(time_points, outlet_temps, label='Outlet Temperature')
plt.xlabel('Time [s]')
plt.ylabel('Temperature [K]')
plt.title('Time Evolution of Outlet Temperature')
plt.legend()

# Spatial profiles for t = 0 and t = 5
z = np.linspace(0, L, N)

plt.subplot(2, 1, 2)
for t_plot, data in spatial_profiles_t.items():
    plt.plot(z, data['T'], label=f'Temperature at t={t_plot:.1f}s')
plt.xlabel('Reactor Length [m]')
plt.ylabel('Temperature [K]')
plt.title('Spatial Profiles')
plt.legend()

plt.tight_layout()
plt.show()
