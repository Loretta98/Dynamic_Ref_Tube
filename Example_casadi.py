# Simple intrgration example 
# Simple optimization example 

from casadi import*

# The single most central functionality of CasADi is algorithmic differetiation (AD)
# 1) Ode solver 
# 2) DAE solver 
# 3) PDE solver 

# Nonlinear programming example 
# min(x2+100z2) s.t. z+(1-x2)+y = 0 

x = SX.sym('x'); y = SX.sym('y'); z = SX.sym('z')
nlp = {'x':vertcat(x,y,z), 'f':x**2+100*z**2, 'g':z+(1-x)**2-y}
# The nlplsol function creates the solver by recalling the built in ipopy solver 
S = nlpsol('S', 'ipopt', nlp)
#print(S)
# First guess for the NLP evaluating function S 
r = S(x0=[2.5,3.0,0.75],\
      lbg=0, ubg=0)
x_opt = r['x']
print('x_opt: ', x_opt)

# The qpsol is supported by qpOASES for resolution of quadratic programming problems 

# First declaration of the variables in a symbolic matter x = SX.sym(), they are all sparse matrixes 
# These then can be used to build symbolic expression for ODE right hand side, algebraic right hand side
# Empty NLP and then we can fill it in 


# ODE initial value problem 
x = MX.sym('x',2); # Two states

# Expression for ODE right-hand side
z = 1-x[1]**2
rhs = vertcat(z*x[0]-x[1],x[0])

ode = {}         # ODE declaration
ode['x']   = x   # states
ode['ode'] = rhs # right-hand side

# Construct a Function that integrates over 4s
F = integrator('F','cvodes',ode,0,4)

# Start from x=[0;1]
res = F(x0=[0,1])

print(res["xf"])

# Sensitivity wrt initial state
res = F(x0=x)
S = Function('S',[x],[jacobian(res["xf"],x)])
print(S([0,1]))