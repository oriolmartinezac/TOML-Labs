from scipy.optimize import minimize
import matplotlib.pyplot as plt
from cvxpy import *
import numpy as np

# Create two scalar optimization variables.
x = Variable(1, name='x')


# Form and solve problem.
prob = Problem(Minimize((square(x[0])+1)), [square(x[0])-4*x[0]+16 <= 0])
print("solve", prob.solve())  # Returns the optimal value.
print ("prob1 is DCP:", prob.is_dcp())
print("status:", prob.status)
print("optimal value p* = ", prob.value)
print("optimal var: x1 = ", x[0].value)
