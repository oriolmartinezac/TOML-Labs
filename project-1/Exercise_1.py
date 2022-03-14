from scipy.optimize import minimize
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import numdifftools as nd

def objective_function():
    return lambda x: (math.e**x[0]) * (4 * ((x[0])**2) + 2 * ((x[1])**2) + 4 * x[0] * x[1] + 1)

def constraints_function():
    return ({'type': 'ineq', 'fun': lambda x: -x[0] * x[1] + x[0] + x[1] - 1.5},
            {'type': 'ineq', 'fun': lambda x: x[0] * x[1] + 10})

def f(x):
    return (math.e**x[0]) * (4 * ((x[0])**2) + 2 * ((x[1])**2) + 4 * x[0] * x[1] + 1)

if __name__ == "__main__":

    x0 = [(0, 0), (10, 20), (-10, 1), (-30, -30)]
    results = []
    x_results = []
    obj_f = objective_function()
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


        #plt.scatter(res.x[0], res.fun, color='orange', marker='x', label='opt1')
        #plt.scatter(res.x[1], res.fun, color='green', marker='x', label='opt2')


    #PLOT IN 3D
    yline = np.arange(-10, 3, 0.7)
    xline = np.arange(-10, 3, 0.7)
    X, Y = np.meshgrid(xline, yline)
    #yline = f((xline, zline))
    Z = np.array(f((X, Y)))
    #yline = [obj_f(val) for val in xline]
    #yline = [obj_f(val) for val in zip(xline, zline)]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)

    #Plot the mins of the different inital guesses
    for i in range(len(results)):
        ax.scatter(x_results[i][0], x_results[i][1], results[i], color="red")

    # Plot a 3D surface
    ax.plot_surface(X, Y, Z)

    plt.show()



