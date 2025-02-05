from casadi import *
import numpy as np
from Input import * 
import matplotlib.pyplot as plt 
from Properties_N import * 
from utils import create_casados_integrator, create_casadi_integrator
from Reactor_model import * 
from scipy.interpolate import interp1d

def plots(profiles,species_labels,t_points,N_t): 
    n_comp = len(species_labels)  # Number of species
    N = profiles.shape[1] // (n_comp + 1)  # Number of spatial points (N)
    times = np.arange(N_t)  # Time steps
    ## **1️⃣ 2D Plot - Inlet Composition (First Spatial Point)**
    plt.figure(figsize=(10, 5))
    for i in range(n_comp):
        plt.plot(t_points, profiles[:, i * N], label=f"{species_labels[i]}")

    plt.xlabel("Time [h]")
    plt.ylabel("Mass Fraction [-]")
    plt.title("Inlet Composition Over Time (x = 0)")
    plt.legend()
    plt.grid()
    ## **1️⃣ 2D Plot - Exit Composition (Last Spatial Point)**
    plt.figure(figsize=(10, 5))
    for i in range(n_comp):
        plt.plot(t_points, profiles[:, (i + 1) * N - 1], label=f"{species_labels[i]}")

    plt.xlabel("Time [h]")
    plt.ylabel("Mole Fraction [-]")
    plt.title("Exit Composition Over Time (Last Spatial Point)")
    plt.legend()
    plt.grid()
    #plt.show()
    ## **3️⃣ 2D Plot - Composition Along Reactor at t = 0.5 h**
    # Find closest time index to t = 0.5 h
    t_target = 0.5
    idx_t = np.argmin(np.abs(np.array(t_points) - t_target))

    plt.figure(figsize=(10, 5))
    x_positions = np.linspace(0, 1, N)  # Assuming normalized reactor length

    for i in range(n_comp):
        plt.plot(x_positions, profiles[idx_t, i * N:(i + 1) * N], label=f"{species_labels[i]}")

    plt.xlabel("Reactor Length [-]")
    plt.ylabel("Mole Fraction [-]")
    plt.title(f"Composition Profile Along Reactor at t = {t_target} h")
    plt.legend()
    plt.grid()
    plt.show()

    ## **2️⃣ 3D Plot - Evolution Over Time for Each Spatial Point**
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i in range(N):  # Loop over spatial points
        for j in range(n_comp):  # Loop over species
            ax.plot(
                t_points[1:],  # Time axis
                [i] * len(t_points[1:]),  # Spatial coordinate (fixed for each line)
                profiles[1:, j * N + i],  # Evolution over time for each spatial point
                label=f"{species_labels[j]} at x_{i}",
                alpha=0.6
            )

    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Spatial Index")
    ax.set_zlabel("Mole Fraction")
    ax.set_title("Species Evolution Over Time & Space")

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Species & Spatial Points")
    plt.show()

def run_steady_sim(integrator_opts, Nsim, plot_traj=True, use_acados=True, use_cython=False):
    u0 = np.array([0.0])
    x0 = vertcat(w0,T0[0])
    dt = L/Nsim
    # create integrator
    model = create_reactor_steady()
    #model = create_reactor_dynamics(Nsim,L)
    if use_acados:
        test_integrator = create_casados_integrator(
            model, integrator_opts, dt=dt, use_cython=use_cython)
        
    else:
        test_integrator = create_casadi_integrator(model, integrator_opts, dt=dt)
    print(f"\n created test_integrator:\n{test_integrator}\n")

    # Store results for plotting
    z_points = [0]  # Start at z = 0
    profiles = [x0.full().flatten()]  # Convert CasADi DM to NumPy array

    # Multiple Shooting
    # Integrate over the spatial domain
    for _ in range(Nsim):
        result = test_integrator(x0=profiles[-1], p=u0)["xf"]
        profiles.append(result.full().flatten())  # Store the result
        z_points.append(z_points[-1] + abs(dt))

    # Convert to arrays for easier handling
    profiles_ = np.array(profiles)
    z_points = np.array(z_points)

    if plot_traj:
        plt.figure(figsize=(10, 6))
        for i in range(profiles_.shape[1]-1):  # Exclude temperature for a separate plot
            plt.plot(z_points, profiles_[:, i], label=f"Component {i}")
        plt.xlabel("Spatial Location (z)")
        plt.ylabel("Mass fraction [-]")
        plt.title("Component Profiles Along the Reactor")
        plt.legend()
        plt.grid()
        
        # Temperature profile
        plt.figure(figsize=(10, 6))
        plt.plot(z_points, profiles_[:, -1]-273.15, label="Temperature", color="red")
        plt.xlabel("Spatial Location (z)")
        plt.ylabel("Temperature (°C)")
        plt.title("Temperature Profile Along the Reactor")
        plt.legend()
        plt.grid()
        #plt.show()
    return profiles,profiles_,x0

def run_dyn_sim(x0,integrator_opts,N,N_t, use_acados=True, use_cython=False): 
    #x0 = np.zeros(N*(n_comp+1)) + (w0_CH4[0], w0_CO[0], w0_CO2[0], w0_H2[0], w0_H2O[0], T0[0])
    #x0 = DM.zeros(N * (n_comp + 1), 1)  # Initialize as a CasADi DM column vector
    #x0[: n_comp + 1] = DM([w0_CH4[0], w0_CO[0], w0_CO2[0], w0_H2[0], w0_H2O[0], T0[0]])  
    x0 = x0
    t_f = 1.0 
    dt = t_f/N_t
    model = create_reactor_dynamics(N,L)
    if use_acados:
        test_integrator = create_casados_integrator(
            model, integrator_opts, dt=dt, use_cython=use_cython)
        
    else:
        test_integrator = create_casadi_integrator(model, integrator_opts, dt=dt)
    print(f"\n created test_integrator:\n{test_integrator}\n")
        # Store results for plotting
    t_points = [0]  # Start at z = 0
    profiles = [x0.full().flatten()]  # Convert CasADi DM to NumPy array
    u0 = 0.0
    # Integrate over the time domain
    for _ in range(N_t):
        result = test_integrator(x0=profiles[-1], p=u0)["xf"]
        profiles.append(result.full().flatten())  # Store the result
        t_points.append(t_points[-1] + abs(dt))
    
    # Plots 

    # Extract species concentrations and temperature
    species_labels = ["CH4", "CO", "CO2", "H2", "H2O"]
    #temperatures = profiles[:, -1]  # Last column is temperature
    profiles = np.array(profiles)
    plots(profiles,species_labels,t_points,N_t)
    
    return ()
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
    Nsim = 250
    integrator_opts = {
        "type": "implicit",
        "collocation_scheme": "radau",
        "num_stages": 9,
        "num_steps": 200,
        "newton_iter": 300,
        "tol": 1e-2,
    }
    N = 80 # space discretization      
    N_t = 100 #time discretization 

    # Get steady-state solution

    steady_state_solution,profiles_,x0 = run_steady_sim(integrator_opts, Nsim)  # Shape: (Nsim, n_comp+1)
      # Convert to CasADi DM if it's a list or NumPy array


    #Reshape `profiles_` to (Nsim, n_comp+1) 
    n_comp_plus_1 = profiles_.size // Nsim  # Infer number of species + 1
    profiles_reshaped = profiles_[:-1].reshape(Nsim, n_comp_plus_1)

    # Define spatial grids
    z_original = np.linspace(0, 1, Nsim)  # Original spatial domain
    z_new = np.linspace(0, 1, N)  # New spatial domain (25 points)

    # Initialize array for interpolated profiles
    profiles_interp = np.zeros((N, n_comp_plus_1))

    # Use cubic interpolation for each species profile
    for i in range(n_comp_plus_1):
        f = interp1d(z_original, profiles_reshaped[:, i], kind="cubic")  
        profiles_interp[:, i] = f(z_new)


    # Flatten & Convert to CasADi format
    x0_reduced = profiles_interp.flatten()
    x0_casadi = DM(x0_reduced)  # Convert to CasADi DM format

    # Convert to CasADi DM and reshape for the dynamic problem
    #x0 = DM(steady_state_solution)
    run_dyn_sim(x0_casadi,integrator_opts,N,N_t) 


if __name__ == "__main__":
    main()

