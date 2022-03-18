import matplotlib.pyplot as plt
from cvxpy import *
import numpy as np

def plot_fun(x):
    return (x**2)+1

# Create one scalar optimization variables.
x = Variable(1, name='x')

obj_fun = square(x[0])+1
cons = square(x[0])-6*x[0]+8
constraints = [cons <= 0]

# Form and solve problem.
prob = Problem(Minimize(obj_fun), constraints)
print("solve", prob.solve())  # Returns the optimal value.
print ("prob1 is DCP:", prob.is_dcp())
print("status:", prob.status)
print("optimal value p* = ", prob.value)
print("optimal var: x1 = ", x[0].value)
print("optimal dual variables lambda1 = ", constraints[0].dual_value)

x_dummy1 = np.linspace(-10, 10, 1000)

y_dummy1 = [plot_fun(val) for val in x_dummy1]
plt.plot(x_dummy1, y_dummy1, color='blue', label='fun')
plt.scatter(x[0].value, prob.value, color='red', marker='x', label='opt')

plt.show()