from scipy.optimize import minimize
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import numdifftools as nd

##TOTALLY CONVEX
def objective_function():
    return lambda x: x[0]**2 + x[1]**2

def f(x):
    return x[0]**2 + x[1]**2

def constraints_function():
    return ({'type': 'ineq', 'fun': lambda x: x[0] - 0.5},
            {'type': 'ineq', 'fun': lambda x: x[0] + x[1] - 1},
            {'type': 'ineq', 'fun': lambda x: x[0]**2 + x[1]**2 - 1},
            {'type': 'ineq', 'fun': lambda x: 9 * (x[0]**2) + x[1]**2 - 9},
            {'type': 'ineq', 'fun': lambda x: x[0]**2 - x[1]},
            {'type': 'ineq', 'fun': lambda x: x[1]**2 - x[0]})

if __name__ == "__main__":
    results = []
    x_results = []
    obj_f = objective_function()
    x0 = [(10, -10), (10, 10)] #First one not feassible and second one feassible

    for element in x0:
        fun = objective_function()
        cons = constraints_function()

        print("NORMAL")
        start_time = time.time()
        res = minimize(fun, element, method='SLSQP', constraints=cons, options={'disp': True})
        print("--- %s seconds ---" % (time.time() - start_time))
        print(res)
        print("optimal value p*", res.fun)
        print("optimal var: x1 = ", res.x[0], " x2 = ", res.x[1])
        results.append(res.fun)
        x_results.append(res.x)
        print("\n")

        print("JACOBIAN")
        g = nd.Gradient(fun)
        start_time = time.time()
        res2 = minimize(fun, element, method='SLSQP', jac=g, constraints=cons, options={'disp': True})
        print("--- %s seconds ---" % (time.time() - start_time))
        print(res2)
        print("optimal value p*", res2.fun)
        print("optimal var: x1 = ", res2.x[0], " x2 = ", res2.x[1])
        print("\n")

    # PLOT IN 3D
    yline = np.arange(-1300, 1300, 100)
    xline = np.arange(-1300, 1300, 100)
    X, Y = np.meshgrid(xline, yline)
    # yline = f((xline, zline))
    Z = np.array(f((X, Y)))
    # yline = [obj_f(val) for val in xline]
    # yline = [obj_f(val) for val in zip(xline, zline)]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)

    # Plot the mins of the different inital guesses
    ax.scatter(x_results[0][0], x_results[0][1], results[0], color="red") #Not feassible
    ax.scatter(x_results[1][0], x_results[1][1], results[1], color="green") #Feassible

    # Plot a 3D surface
    ax.plot_surface(X, Y, Z, cmap="jet")

    plt.show()