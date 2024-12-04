# Resolution of the following single PDE: pi**2 du/dt = d2u/dx2 
# Example from https://uk.mathworks.com/help/matlab/math/solve-single-pde.html
# Domain 0<=x<=1, 
# Boundary conditions: u(x,0) = sin(pi*x), u(0,t) = 0, pi*e-t+du/dx(1,t) = 0
# Full orthogonal discretization method 

from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre
import casadi as ca 

# ----- Collocation points ------
nicp = 1                                # Number of intermediation collocation points

u_results = np.array(u_results)  # Convert list of arrays to 2D NumPy array
# Ensure `X`, `T`, and `u_results` are compatible
X, T = np.meshgrid(x, t_values[:len(u_results)])  # Match time steps

# 3D Surface Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, T, u_results, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('Time (t)')
ax.set_zlabel('u(x, t)')
ax.set_title('Solution of the PDE over time (Orthogonal Collocation)')
fig.colorbar(surface, label='u(x, t)')
plt.show()