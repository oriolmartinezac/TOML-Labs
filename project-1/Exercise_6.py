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

def newtons_method(obj_fun, stop_criteria, max_iterations, inital_guess, **kwargs):
    grad_fun = nd.Gradient(obj_fun)
    hess = nd.Hessian(obj_fun)
    w = inital_guess

    if stop_criteria in kwargs:
        beta = kwargs['stop_criteria']

    weight_history = [w]
    print(obj_fun(w))
    cost_history = [obj_fun(w)]
    for k in range(max_iterations):
        grad_eval = grad_fun(w)
        hess_eval = hess(w)

        hess_eval.shape = (int((np.size(hess_eval))**(0.5)), int((np.size(hess_eval))**(0.5)))

        A = hess_eval + stop_criteria*np.eye(w.size)
        b = grad_eval
        w = np.linalg.solve(A, np.dot(A, w) - b)

        weight_history.append(w)
        cost_history.append(obj_fun(w))

    return weight_history, cost_history

def new_newton(obj_fun, stop_criteria, max_iterations, initial_guess, l_rate):
    grad = nd.Gradient(obj_fun)
    hess = nd.Hessian(obj_fun)
    print("GRAD: ", grad)
    print("HESS: ", hess)
    #np.linalg.solve(hess(x), grad(x))
    #ILL-CONDITIONED
    #np.linalg.cond(H)

    cur_x = initial_guess
    diff = 1.0
    i = 0
    while i < max_iterations and diff > stop_criteria:
        prev_x = cur_x
        #np.linalg.inv(A)

        cur_x = cur_x - l_rate * np.linalg.cond(hess(prev_x)) * grad(prev_x)
        diff = abs(cur_x - prev_x)
        i = i+1
        print("Iteration", i, "\nX value is", cur_x)

    print("The local minimum occurs at", cur_x)
    return cur_x



#obj_fun, stop_criteria, max_iterations, inital_guess
def newton(obj_fun, stop_criteria, max_iterations, initial_guess):
    grad = nd.Gradient(obj_fun)
    hess = nd.Hessian(obj_fun)
    epsilon = stop_criteria
    x = initial_guess
    for i in range(max_iterations):
        x = x - np.linalg.solve(hess(x), grad(x))
        if np.linalg.norm(grad(x)) < epsilon:
            return x, i + 1
    return x, max_iterations

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
        x = gradient_cal(obj_fun2, stop_criteria, max_iterations, element, "backtracking", learning_rate)
        print("\n")

        # PLOT in 2D
        x_dummy1 = np.linspace(-10, 10, 1000)

        y_dummy1 = [plot_fun2(val) for val in x_dummy1]
        plt.plot(x_dummy1, y_dummy1, color='blue', label='fun')
        plt.scatter(x, plot_fun2(x), color='red', marker='x', label='opt')

        plt.show()

    for element in x0:
        print("ITERATION")
        x = new_newton(obj_fun2, stop_criteria, max_iterations, element, learning_rate)
        print("X: ", x)
        print("\n")

        # PLOT in 2D
        x_dummy1 = np.linspace(-10, 10, 1000)

        y_dummy1 = [plot_fun2(val) for val in x_dummy1]
        plt.plot(x_dummy1, y_dummy1, color='blue', label='fun')
        plt.scatter(x, plot_fun2(x), color='red', marker='x', label='opt')

        plt.show()