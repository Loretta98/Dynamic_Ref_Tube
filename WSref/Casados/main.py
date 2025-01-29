from casadi import *
import numpy as np
from Input import * 
import matplotlib.pyplot as plt 
from Properties_N import * 
from utils import create_casados_integrator, create_casadi_integrator
from Reactor_model import * 


def run_forward_sim(integrator_opts,Nsim,dt,vz, plot_traj=True, use_acados=True, use_cython=False):
    u0 = np.array([0.0])
   # x0 = np.zeros(N*(n_comp+1)) + (w0_CH4[0], w0_CO[0], w0_CO2[0], w0_H2[0], w0_H2O[0], T0[0])
    x0 = vertcat(w0)#,T0[0])
    dt = L/Nsim
    # create integrator
    model = create_reactor_steady(Nsim,L,vz)
    #model = create_reactor_dynamics(N,L)
    if use_acados:
        test_integrator = create_casados_integrator(
            model, integrator_opts, dt=dt, use_cython=use_cython)
        
    else:
        test_integrator = create_casadi_integrator(model, integrator_opts, dt=dt)
    print(f"\n created test_integrator:\n{test_integrator}\n")

    # Store results for plotting
    z_points = [0]  # Start at z = 0
    profiles = [x0.full().flatten()]  # Convert CasADi DM to NumPy array

    # Integrate over the spatial domain
    for _ in range(Nsim):
        result = test_integrator(x0=profiles[-1], p=u0)["xf"]
        profiles.append(result.full().flatten())  # Store the result
        z_points.append(z_points[-1] + abs(dt))

    # Convert to arrays for easier handling
    profiles = np.array(profiles)
    z_points = np.array(z_points)

    if plot_traj:
        plt.figure(figsize=(10, 6))
        for i in range(profiles.shape[1]):  # Exclude temperature for a separate plot
            plt.plot(z_points, profiles[:, i], label=f"Component {i}")
        plt.xlabel("Spatial Location (z)")
        plt.ylabel("Concentration")
        plt.title("Component Concentration Profiles Along the Reactor")
        plt.legend()
        plt.grid(True)
        plt.show()
        # # Temperature profile
        # plt.figure(figsize=(10, 6))
        # plt.plot(z_points, profiles[:, -1], label="Temperature", color="red")
        # plt.xlabel("Spatial Location (z)")
        # plt.ylabel("Temperature (K)")
        # plt.title("Temperature Profile Along the Reactor")
        # plt.legend()

    return profiles
    # # print(f"test_integrator.has_jacobian(): {test_integrator.has_jacobian()}")
    # # print(f"test_integrator.jacobian(): {test_integrator.jacobian()}")

    # # test_integrator_sens = test_integrator.jacobian()
    # # print(f"created test_integrator_sens {test_integrator_sens}\n")

    # # open loop simulation
    # simX = np.ndarray((Nsim + 1, nx))
    # simU = np.ndarray((Nsim, nu))
    # x_current = x0
    # simX[0, :] = x_current

    # for i in range(Nsim):
    #     simX[i + 1, :] = (
    #         test_integrator(x0=simX[i, :], p=u0)["xf"].full().reshape((nx,))
    #     )
    #     simU[i, :] = u0

    # # # test call jacobian
    # # sens = test_integrator_sens(x0, u0, x0)
    # # print(f"sens {sens}")

    # results = {"X": simX, "U": simU}

    # print(f"test_forward_sim: SUCCESS!\n")

    # return results

L = 2 

def main(): 
    Nsim = 10 
    vz = 0
    dt = 2
    integrator_opts = {
        "type": "implicit",
        "collocation_scheme": "radau",
        "num_stages": 5,
        "num_steps": 500,
        "newton_iter": 1000,
        "tol": 1e-6,
    }
    results = []
    results = run_forward_sim(integrator_opts,Nsim,dt,vz)

if __name__ == "__main__":
    main()

