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

# Initialize the Solver object.
solver = bacoli_py.Solver()

# Specify the number of PDE's in this system.
npde = 1

# Initialize problem-dependent parameters.
pi = numpy.pi 

# Function defining the PDE system.
# pi^2 * dudt = d2udx2
def f(t, x, u, ux, uxx, fval):
    fval[0] = uxx[0]/(pi**2)
    return fval

# Function defining the left spatial boundary condition.
def bndxa(t, u, ux, bval):
    bval[0] = u[0]
    return bval

# Function defining the right spatial boundary condition.
def bndxb(t, u, ux, bval):
    bval[0] = ux[0] + pi*numpy.exp(-t)
    return bval

# Function defining the initial conditions.
def uinit(x, u):
    u[0] = numpy.sin(pi*x)
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

nt = 10
nx = 10
# Choose output times and points.
tspan = numpy.linspace(0.0001, 2, nt)
xspan = numpy.linspace(0, 1, nx)

# Solve this problem.
evaluation = solver.solve(problem_definition, initial_time, initial_mesh,
                           tspan, xspan, atol=1e-6, rtol=1e-6)

# Plotting these numerical results in 3D.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

styling = {
    'cmap': cm.viridis,
    'linewidth': 0,
    'antialiased': True
}

# Convert xspan and tspan into coordinate arrays for plotting.
T, X = numpy.meshgrid(tspan, xspan)

# Extract the solution for the first PDE in the solved system.
Z = evaluation.u[0, :]

plt.figure()
plt.scatter(xspan,Z[-1])
plt.plot(X,numpy.exp(-2)*numpy.sin(numpy.pi*X))

#fig = plt.figure()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T, X, Z, **styling)

ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_zlabel('$u(t,x)$')

# Show the plot
plt.show()
