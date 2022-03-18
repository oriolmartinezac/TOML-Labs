from scipy.optimize import minimize
import matplotlib.pyplot as plt
from cvxpy import *
import numpy as np
import math

def plot_fun(x):
    return math.log(x[0]) + math.log(x[1]) + math.log(x[2])

# Create two scalar optimization variables.
x = Variable(3, name='x')

c = [1, 2, 1, 2, 1]

obj_fun = cvxpy.log(x[0]) + cvxpy.log(x[1]) + cvxpy.log(x[2])
cons1 = x[0] + x[2]
cons2 = x[0] + x[1]
cons3 = x[2]
cons4 = x[0]
cons5 = x[1]
cons6 = x[2]
constraints = [cons1 <= c[0], cons2 <= c[1], cons3 <= c[4], cons4 >= 0, cons5 >= 0, cons6 >= 0]

# Form and solve problem.
prob = Problem(Maximize(obj_fun), constraints)
print("solve", prob.solve())  # Returns the optimal value.
print ("prob1 is DCP:", prob.is_dcp())
print("status:", prob.status)
print("optimal value p* = ", prob.value)
print("optimal var: x1 = ", x[0].value, " var x2: ", x[1].value, " var x3: ", x[2].value)
