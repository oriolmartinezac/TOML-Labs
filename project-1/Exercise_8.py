from cvxpy import *
import math

def plot_fun(x):
    return math.log(x[0]) + math.log(x[1]) + math.log(x[2])

# Create two scalar optimization variables.
x = Variable(3, name='x')
R = Variable(3, name="R")

obj_fun = cvxpy.sum(cvxpy.log(x))
cons1 = x[0] + x[1]
cons2 = x[0]
cons3 = x[2]
cons4 = R[0] + R[1] + R[2]
constraints = [cons1 <= R[0], cons2 <= R[1], cons3 <= R[2], cons4 <= 1]

# Form and solve problem.
prob = Problem(Maximize(obj_fun), constraints)
print("solve", prob.solve())  # Returns the optimal value.
print("prob1 is DCP:", prob.is_dcp())
print("status:", prob.status)
print("optimal value p* = ", prob.value)
print("optimal var: x1 = ", x[0].value, " var x2: ", x[1].value, " var x3: ", x[2].value)
print("optimal var: R1 = ", R[0].value, " var R2: ", R[1].value, " var R3: ", R[2].value)
print("optimal dual variables lambda1 = ", constraints[0].dual_value)
print("optimal dual variables lambda2 = ", constraints[1].dual_value)
print("optimal dual variables lambda3 = ", constraints[2].dual_value)
print("optimal dual variables lambda4 = ", constraints[3].dual_value)
