import numdifftools as nd
import numpy as np
import matplotlib.pyplot as plt

def obj_fun1():
    return lambda x: 2*(x[0]**2) - 0.5

def plot_fun1(x):
    return 2 * (x ** 2) - 0.5

def obj_fun2():
    return lambda x: 2*(x[0]**4) - 4 * (x[0]**2) + x[0] - 0.5

def plot_fun2(x):
    return 2*(x[0]**4) - 4 * (x[0]**2) + x[0] - 0.5

def gradient_cal(obj_fun, stop_criteria, max_iterations, initial_guess, method, l_rate=0.000001):
    if method == "backtracking":
        cur_x = initial_guess
        i = 0
        diff = 1.0
        grad_fun = nd.Gradient(obj_fun)
        while i < max_iterations and diff > stop_criteria:
            prev_x = cur_x
            cur_x = cur_x - l_rate * grad_fun(prev_x)
            diff = abs(cur_x - prev_x)
            i = i + 1
            print("Iteration", i, "\nX value is", cur_x)

        print("The local minimum occurs at", cur_x)
        return cur_x


if __name__ == "__main__":
    stop_criteria = 0.0001
    max_iterations = 10000
    learning_rate = 0.01

    x0 = 0
    obj_fun1 = obj_fun1()

    x = gradient_cal(obj_fun1, stop_criteria, max_iterations, x0, "backtracking", learning_rate) #1. objective function, 2.stop criteria, 3. max iterations, 4. Inital_guess 5. Method (Backtracking's method, Newton's method)

    # PLOT in 2D
    x_dummy1 = np.linspace(-10, 10, 1000)

    y_dummy1 = [plot_fun1(val) for val in x_dummy1]
    plt.plot(x_dummy1, y_dummy1, color='blue', label='fun')
    plt.scatter(x, plot_fun1(x), color='red', marker='x', label='opt')

    plt.show()

    x0 = [-2, -0.5, 0.5, 2]
    obj_fun2 = obj_fun2()

    for element in x0:
        print("ITERATION")
        x = gradient_cal(obj_fun1, stop_criteria, max_iterations, element, "backtracking", learning_rate)
        print("\n")

        # PLOT in 2D
        x_dummy1 = np.linspace(-10, 10, 1000)

        y_dummy1 = [plot_fun1(val) for val in x_dummy1]
        plt.plot(x_dummy1, y_dummy1, color='blue', label='fun')
        plt.scatter(x, plot_fun1(x), color='red', marker='x', label='opt')

        plt.show()
