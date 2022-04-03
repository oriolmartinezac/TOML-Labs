import numdifftools as nd
import numpy as np
import matplotlib.pyplot as plt

def obj_fun1():
    return lambda x: 2*(x**2) - 0.5

def plot_fun(x, fN=1):
    if fN == 1:
        result = 2 * (x ** 2) - 0.5
    else:
        result = 2 * (x ** 4) - 4 * (x ** 2) + x - 0.5
    return result

def obj_fun2():
    return lambda x: 2*(x**4) - 4 * (x**2) + x - 0.5

def gradient(obj_fun, stop_criteria, max_iterations, initial_guess, l_rate=0.01, fN = 1):
    cur_x = initial_guess
    i = 0
    diff = 1.0
    grad_fun = nd.Gradient(obj_fun)
    plt.scatter(initial_guess, plot_fun(initial_guess, fN), color='yellow', edgecolor='black')

    print("Initial value: ", cur_x)

    while i < max_iterations and diff > stop_criteria:
        prev_x = cur_x
        cur_x = cur_x - l_rate * grad_fun(prev_x)
        diff = abs(cur_x - prev_x)
        i = i + 1
        plt.scatter(cur_x, plot_fun(cur_x, fN), color='green', edgecolor='black')

    print("SOLUTION FOUND")
    print("Number of total iterations", i)
    print("The local minimum occurs at", cur_x)
    return cur_x

def newton(obj_fun, stop_criteria, max_iterations, initial_guess, l_rate=0.01, fN = 1):
    grad = nd.Gradient(obj_fun)
    hess = nd.Hessian(obj_fun)

    cur_x = initial_guess
    diff = 1.0
    i = 0
    plt.scatter(initial_guess, plot_fun(initial_guess, fN), color='yellow', edgecolor='black')

    print("Initial value: ", cur_x)

    while i < max_iterations and diff > stop_criteria:
        prev_x = cur_x
        cur_x = cur_x - l_rate * np.linalg.cond(hess(prev_x)) * grad(prev_x) #ill-conditioned
        diff = abs(cur_x - prev_x)
        i = i+1
        plt.scatter(cur_x, plot_fun(cur_x, fN), color='green', edgecolor='black')

    print("SOLUTION FOUND")
    print("Number of total iterations", i)
    print("The local minimum occurs at", cur_x)
    return cur_x

def solver(obj_fun, stop_criteria, max_iterations, initial_guess, l_rate=0.01, method="", fN=1):
    if method == "backtracking":
        x_result = gradient(obj_fun, stop_criteria, max_iterations, initial_guess, l_rate, fN)
    elif method == "newton":
        x_result = newton(obj_fun, stop_criteria, max_iterations, initial_guess, l_rate, fN)
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
    x = solver(obj_fun1, stop_criteria, max_iterations, x0, learning_rate, "backtracking")  #1. objective function, 2.stop criteria, 3. max iterations, 4. Inital_guess 5. Method (Backtracking's method, Newton's method), 6.Function number to plot the scatter points
    print("\n")

    # PLOT in 2D
    x_dummy1 = np.linspace(-4, 4, 100)

    y_dummy1 = [plot_fun(val) for val in x_dummy1]
    plt.plot(x_dummy1, y_dummy1, color='blue', label='fun')
    plt.scatter(x, plot_fun(x), color='red', marker='x', label='opt')

    plt.show()
    print("NEWTON")
    x = solver(obj_fun1, stop_criteria, max_iterations, x0, learning_rate, "newton")
    print("\n")

    # PLOT in 2D
    x_dummy1 = np.linspace(-4, 4, 100)

    y_dummy1 = [plot_fun(val) for val in x_dummy1]
    plt.plot(x_dummy1, y_dummy1, color='blue', label='fun')
    plt.scatter(x, plot_fun(x), color='red', marker='x', label='opt')

    plt.show()

    x0 = [-2, -0.5, 0.5, 2]
    obj_fun2 = obj_fun2()

    for element in x0:
        print("BACKTRACKING")
        x = solver(obj_fun2, stop_criteria, max_iterations, element, learning_rate, "backtracking", 2)
        print("\n")

        # PLOT in 2D
        x_dummy1 = np.linspace(-2, 2, 100)

        y_dummy1 = [plot_fun(val, 2) for val in x_dummy1]
        plt.plot(x_dummy1, y_dummy1, color='blue', label='fun')
        plt.scatter(x, plot_fun(x, 2), color='red', marker='x', label='opt')

        plt.show()

        print("NEWTON")
        x = solver(obj_fun2, stop_criteria, max_iterations, element, learning_rate, "newton", 2)
        print("X: ", x)
        print("\n")

        # PLOT in 2D
        x_dummy1 = np.linspace(-2, 2, 100)

        y_dummy1 = [plot_fun(val, 2) for val in x_dummy1]
        plt.plot(x_dummy1, y_dummy1, color='blue', label='fun')
        plt.scatter(x, plot_fun(x, 2), color='red', marker='x', label='opt')

        plt.show()