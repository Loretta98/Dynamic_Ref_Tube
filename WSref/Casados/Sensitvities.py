import numpy as np
import matplotlib.pyplot as plt
from utils import create_casados_integrator
from Reactor_model import create_reactor_steady
from casadi import * 
from Input import w0

def test_integrator_sensitivity(model, integrator_opts, dt_values, param_sets, use_cython=False):
    """
    Test the sensitivity of the integrator by varying key parameters and plotting results.
    """
    results = {"Time Step": []}
    
    reference_opts = integrator_opts.copy()
    # for param_name, values in param_sets:
    #     reference_opts[param_name.lower().replace(" ", "_")] = max(values)
    x0 = vertcat(w0)#,T0[0])
    # Create reference solution with highest accuracy settings
    
    reference_integrator = create_casados_integrator(model, reference_opts, dt = dt_values[0], use_cython=use_cython)
        # Integrate over the spatial domain
    
    profiles = [x0.full().flatten()]
    Nsim = 10 
    u0 = np.array([0.0])

    for _ in range(Nsim):
        ref_solution = reference_integrator(x0=profiles[-1], p=u0)["xf"]
        profiles.append(ref_solution.full().flatten())  # Store the result
    
    print(f"Reference solution: {profiles}")

    dt = dt_values
    # Test other parameter variations
    for param_name, values in param_sets:
        results[param_name] = []
        for val in values:
            test_opts = integrator_opts.copy()
            test_opts[param_name.lower().replace(" ", "_")] = val
            
            integrator = create_casados_integrator(model, test_opts, max(dt_values), use_cython)
            solution = integrator(x0=x0, p=np.array([0.0]))["xf"].full().flatten()
            print(f"Reference solution: {ref_solution}")
            print(f"Test solution for dt={dt}: {solution}")
            print(f"Solution for dt={dt}: {solution}")
            if np.any(np.isnan(solution)) or np.all(solution == 0):
                print(f"Warning: Solution for dt={dt} is invalid")

            deviation = np.linalg.norm(solution - ref_solution) / np.linalg.norm(ref_solution)
            results[param_name].append((val, deviation))

    # Plot results
    fig, axs = plt.subplots(1, len(param_sets) + 1, figsize=(15, 5))
    
    for j, (param_name, values) in enumerate(param_sets):
        vals, deviations = zip(*results[param_name])
        axs[j].plot(vals, deviations, marker='o', linestyle='--')
        axs[j].set_xlabel(param_name)
        axs[j].set_ylabel("Relative Deviation from Reference")
        axs[j].set_title(f"Sensitivity to {param_name}")
        axs[j].grid()
    
    # Plot dt sensitivity
    dt_vals, dt_devs = zip(*results["Time Step"])
    axs[-1].plot(dt_vals, dt_devs, marker='o', linestyle='--')
    axs[-1].set_xlabel("Time Step (dt)")
    axs[-1].set_ylabel("Relative Deviation from Reference")
    axs[-1].set_title("Sensitivity to Time Step")
    axs[-1].grid()
    
    plt.tight_layout()
    plt.show()

def main():
    L = 2  # Reactor length
    Nsim_values = [10]  # Different Nsim values to test
    vz = 0  
    dt_values = [L / Nsim for Nsim in Nsim_values]  # Compute dt based on Nsim

    param_sets = [
        ("Tolerance", [1e-6]),
        ("Num Steps", [50,200,600]),
        ("Num Stages", [2,6,9]),  # Cannot go above 9 for Radau
        ("Newton Iterations", [1000])
    ]
    
    for Nsim in Nsim_values:
        model = create_reactor_steady(Nsim, L, vz)
        
        integrator_opts = {
            "type":"implicit",
            "collocation_scheme": "radau",
            "tol": 1e-6,
            "num_steps": 500,
            "num_stages": 9,
            "newton_iter": 1000
        }
        
        test_integrator_sensitivity(model, integrator_opts, dt_values, param_sets, use_cython=False)

if __name__ == "__main__":
    main()
