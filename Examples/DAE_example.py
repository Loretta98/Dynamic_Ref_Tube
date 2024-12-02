# Resolution of a DAE problem for a dynamic PFR 
# Setup of a DAE probelm using the DAE builder class from github.com/casadi


from casadi import *

# Example on how to use the DaeBuilder class
# Joel Andersson, UW Madison 2017

# Start with an empty DaeBuilder instance
dae = DaeBuilder('rocket')

# Add input expressions
a = dae.add_p('a')
b = dae.add_p('b')
u = dae.add_u('u')
h = dae.add_x('h')
v = dae.add_x('v')
m = dae.add_x('m')

# Constants
g = 9.81 # gravity

# Set ODE right-hand-side
dae.set_ode('h', v)
dae.set_ode('v', (u-a*v**2)/m-g)
dae.set_ode('m', -b*u**2)

# Specify initial conditions
dae.set_start('h', 0)
dae.set_start('v', 0)
dae.set_start('m', 1)

# Add meta information
dae.set_unit('h','m')
dae.set_unit('v','m/s')
dae.set_unit('m','kg')

# Print DAE
#dae.disp(True)

# Integrator for the DAE, it doesn't work with the DAEbuilder, the dae must be setup with symbolic framework. 
F = integrator('F','idas',dae)
print(F)