from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt
import numpy as np
import time
def objective_function():
    return lambda x: (math.e**x[0]) * (4 * ((x[0])**2) + 2 * ((x[1])**2) + 4 * x[0] * x[1] + 1)

def constraints_function():
    return ({'type': 'ineq', 'fun': lambda x: -x[0] * x[1] + x[0] + x[1] - 1.5},
            {'type': 'ineq', 'fun': lambda x: x[0] * x[1] + 10})



def f(x):
    return (math.e**x[0]) * (4 * ((x[0])**2) + 2 * ((x[1])**2) + 4 * x[0] * x[1] + 1)

def fun_Jac(x):
    dx = 2*x[0]
    dy = 2*x[1]
    return np.array((dx, dy))


if __name__ == "__main__":

    x0 = [(0, 0), (10, 20), (-10, 1), (-30, -30)]

    for element in x0:
        fun = objective_function()
        cons = constraints_function()

        print("NORMAL")
        start_time = time.time()
        res = minimize(fun, element, method='SLSQP', constraints=cons)
        print("--- %s seconds ---" % (time.time() - start_time))
        print(res)
        print("optimal value p*", res.fun)
        print("optimal var: x1 = ", res.x[0], " x2 = ", res.x[1])
        print("\n")

        print("JACOBIAN")
        start_time = time.time()
        res2 = minimize(fun, fun_Jac(element), method='SLSQP', constraints=cons)
        print("--- %s seconds ---" % (time.time() - start_time))
        print(res)
        print("optimal value p*", res.fun)
        print("optimal var: x1 = ", res.x[0], " x2 = ", res.x[1])

        print("\n")

        #x_dummy1 = np.linspace(res.x, res.x, 1000)
        #y_dummy1 = [fun(val) for val in x_dummy1]
        #plt.plot(x_dummy1, y_dummy1, color='blue', label='fun')
        #plt.scatter(res.x, res.fun, color='orange', marker='x', label='opt')
        #x_dummy = np.linspace(start=-1000, stop=100, num=10000)
        #y_dummy = n
        #plt.plot(x_dummy, x_dummy**2)
        #plt.plot(res.x[0], f(res.x), 'ro')

        #plt.grid()
        #plt.legend(loc=1)
        #plt.show()





