from casados_integrator import CasadosIntegrator
from acados_template import AcadosModel, AcadosSimSolver, AcadosSim
import casadi as ca

def create_casados_integrator2(dae, ts, collocation_opts=None, record_time=False, with_sensitivities=True, use_cython=True):

    # dimensions
    nx = dae.size1_in(1)
    nu = dae.size1_in(2)
    nz = dae.size1_in(3)

    # create acados model
    model = AcadosModel()
    model.x = ca.MX.sym('x',nx)
    model.u = ca.MX.sym('u',nu)
    model.z = ca.MX.sym('z', nz)
    model.xdot = ca.MX.sym('xdot', nx)
    model.p = []
    model.name = 'reactor_model'

    # f(xdot, x, u, z) = 0
    model.f_impl_expr = dae(model.xdot, model.x, model.u[:nu], model.z)

    sim = AcadosSim()
    sim.model = model
    sim.solver_options.T = ts
    sim.solver_options.integrator_type = 'IRK'
    sim.solver_options.num_stages = 4 # nlp.d
    sim.solver_options.num_steps = 1
    # sim.solver_options.newton_iter = 20
    sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"
    if with_sensitivities:
        sim.solver_options.sens_forw = True
        sim.solver_options.sens_algebraic = False
        sim.solver_options.sens_hess = True
        sim.solver_options.sens_adj = True

    if collocation_opts is not None:
        # sim.solver_options.T = collocation_opts['tf']
        sim.solver_options.num_steps = collocation_opts['number_of_finite_elements']
        sim.solver_options.num_stages = collocation_opts['interpolation_order']
        if collocation_opts['collocation_scheme'] == 'radau':
            sim.solver_options.collocation_type = 'GAUSS_RADAU_IIA'
        sim.solver_options.newton_tol = collocation_opts['rootfinder_options']['abstolStep']
        sim.solver_options.newton_iter = collocation_opts['rootfinder_options']['max_iter']

    function_opts = {"record_time": record_time}

    casados_integrator = CasadosIntegrator(sim, use_cython=use_cython)
    # reformat for tunempc
    x = ca.MX.sym('x', nx)
    u = ca.MX.sym('u', nu)
    xf = casados_integrator(x, u)
    f = ca.Function('f', [x,u], [xf], ['x0','p'], ['xf'], function_opts)
    l = ca.Function('l', [x,u], [xf[-1]], function_opts)

    return casados_integrator, f, l

def create_casados_integrator(model, integrator_opts, dt=0.1, use_cython=True, integrator_type="IRK", code_reuse=False):
    sim = AcadosSim()
    sim.model = model

    # set simulation time
    sim.solver_options.T = dt

    # set options

    sim.solver_options.sens_forw = True
    sim.solver_options.sens_algebraic = False
    sim.solver_options.sens_hess = True
    sim.solver_options.sens_adj = True
    # if integrator_opts["type"] == "implicit":
    if integrator_type == "GNSF":
        sim.solver_options.integrator_type = "GNSF"
        sim.solver_options.sens_hess = False
    elif integrator_type == "RK4":
        sim.solver_options.integrator_type = "ERK"
    else:
        sim.solver_options.integrator_type = "IRK"
    # elif integrator_opts["type"] == "explicit":
    #     sim.solver_options.integrator_type = "ERK"
    # else:
    #     raise Exception("integrator_opts['type'] must be explicit or implicit.")

    if integrator_type == "RK4":
        sim.solver_options.num_stages = 4
        sim.solver_options.num_steps = 1
    else:
        sim.solver_options.num_stages = integrator_opts["num_stages"]
        sim.solver_options.num_steps = integrator_opts["num_steps"]

    sim.solver_options.newton_iter = integrator_opts["newton_iter"]

    if integrator_opts["collocation_scheme"] == "radau":
        sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"
    elif integrator_opts["collocation_scheme"] == "legendre":
        sim.solver_options.collocation_type = "GAUSS_LEGENDRE"
    else:
        raise Exception(
            "integrator_opts['collocation_scheme'] must be radau or legendre."
        )

    sim.solver_options.newton_tol = (
        integrator_opts["tol"] / integrator_opts["num_steps"]
    )
    sim.code_export_directory = f'c_generated_code_{model.name}_{sim.solver_options.integrator_type}'

    # create
    casados_integrator = CasadosIntegrator(sim, use_cython=use_cython, code_reuse=code_reuse)

    # if integrator_opts['type'] == 'implicit':
    #     casados_integrator.acados_integrator.set('xdot', np.zeros(casados_integrator.nx))

    return casados_integrator

def create_casados_integrator1(dae, ts, collocation_opts=None, record_time=False, with_sensitivities=True, use_cython=True):

    # dimensions
    nx = dae.size1_in(1)
    nz = dae.size1_in(2)
    nu = dae.size1_in(3)

    # create acados model
    model = AcadosModel()
    model.x = ca.MX.sym('x',nx)
    model.u = ca.MX.sym('u',nu)
    model.z = ca.MX.sym('z', nz)
    model.xdot = ca.MX.sym('xdot', nx)
    model.p = []
    model.name = 'reactor_model'

    # f(xdot, x, u, z) = 0
    model.f_impl_expr = dae(model.xdot, model.x, model.u[:nu], model.z)

    sim = AcadosSim()
    sim.model = model
    sim.solver_options.T = ts
    sim.solver_options.integrator_type = 'IRK'
    sim.solver_options.num_stages = 4 # nlp.d
    sim.solver_options.num_steps = 1
    # sim.solver_options.newton_iter = 20
    sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"
    if with_sensitivities:
        sim.solver_options.sens_forw = True
        sim.solver_options.sens_algebraic = False
        sim.solver_options.sens_hess = True
        sim.solver_options.sens_adj = True

    if collocation_opts is not None:
        # sim.solver_options.T = collocation_opts['tf']
        sim.solver_options.num_steps = collocation_opts['number_of_finite_elements']
        sim.solver_options.num_stages = collocation_opts['interpolation_order']
        if collocation_opts['collocation_scheme'] == 'radau':
            sim.solver_options.collocation_type = 'GAUSS_RADAU_IIA'
        sim.solver_options.newton_tol = collocation_opts['rootfinder_options']['abstolStep']
        sim.solver_options.newton_iter = collocation_opts['rootfinder_options']['max_iter']

    function_opts = {"record_time": record_time}

    casados_integrator = CasadosIntegrator(sim, use_cython=use_cython)
    # reformat for tunempc
    x = ca.MX.sym('x', nx)
    u = ca.MX.sym('u', nu)
    xf = casados_integrator(x, u)
    f = ca.Function('f', [x,u], [xf], ['x0','p'], ['xf'], function_opts)
    l = ca.Function('l', [x,u], [xf[-1]], function_opts)

    return casados_integrator, f, l


