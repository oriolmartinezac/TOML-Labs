import numdifftools as nd
import numpy as np
import matplotlib.pyplot as plt

from autograd import grad
from autograd import hessian

def obj_fun1():
    return lambda x: 2*(x[0]**2) - 0.5

def plot_fun1(x):
    return 2 * (x ** 2) - 0.5

def obj_fun2():
    return lambda x: 2*(x**4) - 4 * (x**2) + x - 0.5

def plot_fun2(x):
    return 2*(x**4) - 4 * (x**2) + x - 0.5

def gradient(obj_fun, stop_criteria, max_iterations, initial_guess, l_rate=0.01):
    cur_x = initial_guess
    i = 0
    diff = 1.0
    grad_fun = nd.Gradient(obj_fun)

    print("Initial value: ", cur_x)

    while i < max_iterations and diff > stop_criteria:
        prev_x = cur_x
        cur_x = cur_x - l_rate * grad_fun(prev_x)
        diff = abs(cur_x - prev_x)
        i = i + 1

    print("SOLUTION FOUND")
    print("Number of total iterations", i)
    print("The local minimum occurs at", cur_x)
    return cur_x

def newton(obj_fun, stop_criteria, max_iterations, initial_guess, l_rate=0.01):
    grad = nd.Gradient(obj_fun)
    hess = nd.Hessian(obj_fun)

    cur_x = initial_guess
    diff = 1.0
    i = 0

    print("Initial value: ", cur_x)

    while i < max_iterations and diff > stop_criteria:
        prev_x = cur_x
        cur_x = cur_x - l_rate * np.linalg.cond(hess(prev_x)) * grad(prev_x) #ill-conditioned
        diff = abs(cur_x - prev_x)
        i = i+1

    print("SOLUTION FOUND")
    print("Number of total iterations", i)
    print("The local minimum occurs at", cur_x)
    return cur_x

def solver(obj_fun, stop_criteria, max_iterations, initial_guess, l_rate=0.01, method=""):
    if method == "backtracking":
        x_result = gradient(obj_fun, stop_criteria, max_iterations, initial_guess, l_rate)
    elif method == "newton":
        x_result = newton(obj_fun, stop_criteria, max_iterations, initial_guess, l_rate)
    else:
        x_result = None
    return x_result

if __name__ == "__main__":

    stop_criteria = 0.0001
    max_iterations = 10000
    learning_rate = 0.01

    x0 = 3
    obj_fun1 = obj_fun1()
    print("BACKTRACKING")
    x = solver(obj_fun1, stop_criteria, max_iterations, x0, learning_rate, "backtracking")  #1. objective function, 2.stop criteria, 3. max iterations, 4. Inital_guess 5. Method (Backtracking's method, Newton's method)
    print("\n")

    # PLOT in 2D
    x_dummy1 = np.linspace(-10, 10, 1000)

    y_dummy1 = [plot_fun1(val) for val in x_dummy1]
    plt.plot(x_dummy1, y_dummy1, color='blue', label='fun')
    plt.scatter(x, plot_fun1(x), color='red', marker='x', label='opt')

    plt.show()
    print("NEWTON")
    x = solver(obj_fun1, stop_criteria, max_iterations, x0, learning_rate, "newton")
    print("\n")

    # PLOT in 2D
    x_dummy1 = np.linspace(-10, 10, 1000)

    y_dummy1 = [plot_fun1(val) for val in x_dummy1]
    plt.plot(x_dummy1, y_dummy1, color='blue', label='fun')
    plt.scatter(x, plot_fun1(x), color='red', marker='x', label='opt')

    plt.show()

    x0 = [-2, -0.5, 0.5, 2]
    obj_fun2 = obj_fun2()

    for element in x0:
        print("BACKTRACKING")
        x = solver(obj_fun2, stop_criteria, max_iterations, element, learning_rate, "backtracking")
        print("\n")

        # PLOT in 2D
        x_dummy1 = np.linspace(-10, 10, 1000)

        y_dummy1 = [plot_fun2(val) for val in x_dummy1]
        plt.plot(x_dummy1, y_dummy1, color='blue', label='fun')
        plt.scatter(x, plot_fun2(x), color='red', marker='x', label='opt')

        plt.show()

        print("NEWTON")
        x = solver(obj_fun2, stop_criteria, max_iterations, element, learning_rate, "newton")
        print("X: ", x)
        print("\n")

        # PLOT in 2D
        x_dummy1 = np.linspace(-10, 10, 1000)

        y_dummy1 = [plot_fun2(val) for val in x_dummy1]
        plt.plot(x_dummy1, y_dummy1, color='blue', label='fun')
        plt.scatter(x, plot_fun2(x), color='red', marker='x', label='opt')

        plt.show()