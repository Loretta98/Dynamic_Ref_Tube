from casadi import *
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import casadi 

# Define Cantera model
mech = "../data/SiF4_NH3_mec.yaml"
gas, bulk_Si, bulk_N = ct.import_phases(mech, ["gas", "SiBulk", "NBulk"])
gas_Si_N_interface = ct.Interface(mech, "SI3N4", [gas, bulk_Si, bulk_N])

# Operating conditions
T0 = 1713  # Kelvin
p0 = 2 * ct.one_atm / 760.0  # Pa ~2 Torr
gas.TPX = T0, p0, "NH3:6, SiF4:1"
bulk_Si.TP = T0, p0
bulk_N.TP = T0, p0
gas_Si_N_interface.TP = T0, p0
gas_Si_N_interface.coverages = "HN_NH2(S): 1.0"
gas_Si_N_interface.advance_coverages(100.0)  # Steady state for surface species

# Reactor dimensions
D = 5.08e-2  # m
Ac = np.pi * D**2 / 4  # m^2
perim = np.pi * D
mu = 5.7e-5  # Pa.s

# Initial state
N = gas.n_species
M = gas_Si_N_interface.n_species
u0 = 11.53  # m/s
rho0 = gas.density
Yk0 = gas.Y
p0 = gas.P
Zk0 = gas_Si_N_interface.coverages

# Define CasADi variables
u = SX.sym("u")  # Velocity
rho = SX.sym("rho")  # Density
Yk = SX.sym("Yk", N)  # Mass fractions
p = SX.sym("p")  # Pressure
Zk = SX.sym("Zk", N)  # Surface fractions

# Spatial derivatives
du_dz = SX.sym("du_dz")
drho_dz = SX.sym("drho_dz")
dYk_dz = SX.sym("dYk_dz", N)
dp_dz = SX.sym("dp_dz")
vec = vertcat(u, rho, Yk, p, Zk)
vecp = vertcat(du_dz, drho_dz, dYk_dz, dp_dz)

# Residual equations
sdot_g = SX.sym("sdot_g", N)  # Gas surface production rates
wdot_g = SX.sym("wdot_g", N)  # Gas homogeneous production rates
sdot_s = SX.sym("sdot_s", M)  # Surface production rates

result = []
result.append(u * drho_dz + rho * du_dz - perim * dot(sdot_g, gas.molecular_weights) / Ac)
for k in range(N):
    result.append(
        rho * u * Ac * dYk_dz[k]
        + Yk[k] * perim * dot(sdot_g, gas.molecular_weights)
        - wdot_g[k] * gas.molecular_weights[k] * Ac
        - sdot_g[k] * gas.molecular_weights[k] * perim
    )
result.append(2 * rho * u * du_dz + u**2 * drho_dz + dp_dz + 32 * u * mu / D**2)
result.append(rho - p * gas.mean_molecular_weight / (ct.gas_constant * T0))
for j in range(M):
    result.append(sdot_s[j])
# Replace sum(Zk) = 1 constraint for the dominant site fraction
for j in range(M):
    result[3 + N + j] = casadi.sum1(Zk) - 1

# Define the DAE function
residual = vertcat(*result)
dae_function = Function("dae", [vec, vecp, sdot_g, wdot_g, sdot_s], [residual])
inputs = casadi.vertcat(u, rho, Yk, Zk)  # Combine all inputs
J = casadi.jacobian(inputs, inputs)  # Compute Jacobian
if casadi.det(J) == 0:
    print("Inputs are not independent")
# Solver setup
dae = {"x": vec, "z": Zk, "p": vertcat(sdot_g, wdot_g, sdot_s), "ode": vecp, "alg": residual}
opts = {"tf": 1.0, "abstol": 1e-8, "reltol": 1e-6}
solver = integrator("solver", "idas", dae, opts)

# Initial conditions
vec0 = vertcat(u0, rho0, Yk0, p0, Zk0)
vecp0 = vertcat(np.zeros_like(vec0))

# Solve
solution = solver(x0=vec0, z0=Zk0, p=vertcat(sdot_g, wdot_g, sdot_s))

# Extract results
times = np.linspace(0, 1.0, 100)
states = solution["xf"].full()

# Plot results
plt.plot(times, states[:, 0])  # Example plot for velocity
plt.xlabel("Distance (m)")
plt.ylabel("Velocity (m/s)")
plt.show()
