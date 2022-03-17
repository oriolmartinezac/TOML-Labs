from scipy.optimize import minimize
import matplotlib.pyplot as plt
from cvxpy import *
import numpy as np

def plot_fun(x):
    return x[0]**2+x[1]**2

# Create two scalar optimization variables.
x = Variable(2, name='x')

obj_fun = square(x[0])+square(x[1])
cons1 = square(x[0]-1)+square(x[1]-1)
cons2 = square(x[0]-1)+square(x[1]+1)
constraints = [cons1 <= 1, cons2 <= 1]

# Form and solve problem.
prob = Problem(Minimize(obj_fun), constraints)
print("solve", prob.solve())  # Returns the optimal value.
print ("prob1 is DCP:", prob.is_dcp())
print("status:", prob.status)
print("optimal value p* = ", prob.value)
print("optimal var: x1 = ", x[0].value, x[1].value)

#PLOT IN 3D
yline = np.arange(-10, 10, 0.7)
xline = np.arange(-10, 10, 0.7)
X, Y = np.meshgrid(xline, yline)
Z = np.array(plot_fun((X, Y)))
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)

#Plot the mins of the different inital guesses
ax.scatter(x[0].value, x[1].value, prob.value, color="red")

# Plot a 3D surface
ax.plot_surface(X, Y, Z, cmap="cool")

plt.show()