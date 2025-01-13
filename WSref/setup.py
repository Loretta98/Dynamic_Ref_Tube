from acados_template import AcadosOcp, AcadosOcpSolver

def setup_acados_integrator(dynamics, T_sim, integration_method="ERK"):
    """
    Set up an ACADOS integrator using the given system dynamics.
    """
    # Create ACADOS OCP object
    ocp = AcadosOcp()

    # Assign model and dynamics
    model = dynamics
    ocp.model = model

    # Set integration parameters
    ocp.dims.N = 1  # Single-step integrator
    ocp.solver_options.integrator_type = integration_method  # 'ERK', 'IRK', etc.
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1

    # Define time step
    ocp.solver_options.Tsim = T_sim

    # Create the integrator solver
    integrator = AcadosOcpSolver(ocp, json_file="acados_ocp.json")
    return integrator
