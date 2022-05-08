import random

from header import *  # IMPORTING HEADER FILE
import matplotlib.pyplot as plt
import gpkit
import cvxpy
import gpkit.nomials

def calc_n_d(d):
    n_d = (2 * d - 1) * C
    if d == 0:
        n_d = 1
    return n_d


def calc_i_d(d):
    if d == 0:
        i_d = C
    elif d == D:
        i_d = 0
    else:
        i_d = (2 * d + 1) / (2 * d - 1)
    return i_d


def calc_f_out(d):
    f_out = Fs * ((D ** 2 - (d ** 2) + 2 * d - 1) / (2 * d - 1))
    if d == D:
        f_out = Fs
    return f_out


def calc_f_b(d):
    return (C - abs(calc_i_d(d))) * calc_f_out(d)


def calc_f_i(d):
    f_i = Fs * ((D ** 2 - (d ** 2)) / (2 * d - 1))
    if d == 0:
        f_i = Fs * (D ** 2) * C
    return f_i


def calc_alphas(d):
    alpha1 = Tcs + Tal + 3 / 2 * Tps * ((Tps + Tal) / 2 + Tack + Tdata) * calc_f_b(d)
    alpha2 = calc_f_out(d) / 2
    alpha3 = ((Tps + Tal) / 2 + Tcs + Tal + Tack + Tdata) * calc_f_out(d) + (
            3 / 2 * Tps + Tack + Tdata) * calc_f_i(d) + 3 / 4 * Tps * calc_f_b(d)

    return alpha1, alpha2, alpha3


def calc_betas(d):
    beta1 = sum([1 / 2] * d)
    beta2 = sum([Tcw / 2 + Tdata] * d)

    return beta1, beta2


def energy_fun(tw):  # ENERGY FUNCTION
    return alpha_1 / tw + alpha_2 * tw + alpha_3


def delay_fun(tw):
    return beta_1 * tw + beta_2


if __name__ == "__main__":
    # PART 1 #
    time1 = [1, 5, 10, 15, 20, 25, 30]
    time = [5, 10, 15, 20, 25]
    alpha_1, alpha_2, alpha_3 = 0.0, 0.0, 0.0
    x = np.linspace(Tw_min, Tw_max)

    for t in time1:
        Fs = 1.0 / (t * 60 * 1000)
        alpha_1, alpha_2, alpha_3 = calc_alphas(1)
        label = "Fs= "+str(t)
        plt.plot(x, energy_fun(x), label=label)
        plt.xlabel('Tw')
        plt.ylabel('Energy consumed')
        plt.title("Energy function of Tw with different Fs")

    plt.legend()
    plt.savefig("1-energy.jpg")
    plt.show()

    beta_1, beta_2 = calc_betas(D)
    plt.plot(x, delay_fun(x), color='red', label='fun')
    plt.xlabel('Tw')
    plt.ylabel('Delay time')
    plt.title("Delay function of Tw")
    plt.savefig("1-delay.jpg")
    plt.show()

    x = np.linspace(10, Tw_max)

    for t in time1:
        Fs = 1.0 / (t * 60 * 1000)
        alpha_1, alpha_2, alpha_3 = calc_alphas(1)
        beta_1, beta_2 = calc_betas(D)
        label = "Fs= "+str(t)
        plt.plot(energy_fun(x), delay_fun(x), label=label)
        plt.xlabel('Energy')
        plt.ylabel('Delay')
        plt.title("Energy-Delay")
    plt.legend()
    plt.savefig("1-energy-delay.jpg")
    plt.show()

    # PART 2 #
    prob1_solves = []
    np_L = np.linspace(100, 5000, 50)
    list_Lmax = [500, 750, 1000, 2500, 5000]
    tw_np = np.linspace(Tw_min, Tw_max)

    colours_plot = ['blue', 'purple', 'green', 'yellow', 'black']
    size_colours = len(colours_plot)
    colour_index = 0

    for l_element in list_Lmax:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for t in time:
            x = gpkit.Variable("x")

            Fs = 1.0 / (t * 60 * 1000)
            alpha_1, alpha_2, alpha_3 = calc_alphas(1)
            beta_1, beta_2 = calc_betas(D)
            Tt_x = (x / (Tps + Tal)) * ((Tps + Tal) / 2) + Tack + Tdata
            E1_tx = (Tcs + Tal + Tt_x) * calc_f_out(1)

            obj_fun1 = alpha_1 / x + alpha_2 * x + alpha_3
            cons1 = beta_1 * x + beta_2
            cons2 = x
            cons3 = abs(calc_i_d(0)) * E1_tx
            constraints = [cons1 <= l_element, cons2 >= Tw_min, cons3 <= 1 / 4]
            prob1 = gpkit.Model(obj_fun1, constraints)
            solution = prob1.solve()
            print(solution['variables']['x'], solution['cost'])
            prob1_solves.append(solution["cost"])
            plt.plot(tw_np, energy_fun(tw_np), color=colours_plot[colour_index % size_colours], label='E(Tw) for Fs('+str(t)+'min)')
            colour_index += 1
            ax.scatter(solution['variables'][x], solution['cost'], color="red")

        plt.xlabel('Tw (ms)')
        plt.ylabel('Energy (J)')
        plt.legend(loc='upper right')
        plt.title("L_max="+str(l_element))
        plt.savefig("2-"+str(l_element)+".jpg")
        plt.show()

    colour_index = 0
    list_Ebudget = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    prob2_solves = []

    for e_element in list_Ebudget:
        x = gpkit.Variable("x")
        Fs = 1.0 / (5 * 60 * 1000)
        alpha_1, alpha_2, alpha_3 = calc_alphas(1)
        beta_1, beta_2 = calc_betas(D)
        Tt_x = (x / (Tps + Tal)) * ((Tps + Tal) / 2) + Tack + Tdata
        E1_tx = (Tcs + Tal + Tt_x) * calc_f_out(1)

        obj_fun2 = beta_1 * x + beta_2
        cons1 = alpha_1 / x + alpha_2 * x + alpha_3
        cons2 = x
        cons3 = abs(calc_i_d(0)) * E1_tx
        constraints = [cons1 <= e_element, cons2 >= Tw_min, cons3 <= (1 / 4)]
        prob2 = gpkit.Model(obj_fun2, constraints)
        solution = prob2.solve()
        print(solution['variables'][x], solution['cost'])
        prob2_solves.append(solution["cost"])
        plt.plot(tw_np, delay_fun(tw_np), color=colours_plot[colour_index % size_colours])
        colour_index += 1
        ax.scatter(solution['variables'][x], solution['cost'], color="red")


    plt.xlabel('Tw (ms)')
    plt.ylabel('Delay')
    plt.title("All Ebudgets")
    plt.savefig("allebudgets.jpg")
    plt.show()

    """
    #Previous E_BUDGET for exercise 2
    prob2_solves = []
    np_E = np.linspace(0.5, 5, 50)

    for e_item in np_E:
        obj_fun2 = beta_1 * x + beta_2
        cons1 = alpha_1 / x + alpha_2 * x + alpha_3
        cons2 = x
        cons3 = abs(calc_i_d(0)) * E1_tx
        constraints = [cons1 <= e_item, cons2 >= Tw_min, cons3 <= (1 / 4)]
        prob2 = gpkit.Model(obj_fun2, constraints)
        solution = prob2.solve()
        print(solution['variables']['x'], solution['cost'])
        prob2_solves.append([solution['variables']['x'], solution["cost"]])

    plt.plot(np_E, prob2_solves, color="blue")
    plt.xlabel('E_budget')
    plt.ylabel('Minimization')
    plt.title("Problem 2 to optimize")
    plt.show()

    plt.plot(energy_fun(np_E), delay_fun(np_L), color="green")
    plt.xlabel('Energy')
    plt.ylabel('Delay')
    plt.title("Energy-Delay function")
    plt.show()
    """
    # PART 3 #
    # Game

    colours = ['red', 'green', 'yellow', 'black', 'purple', 'orange']
    size_colours = len(colours)
    Tw_n = np.linspace(50, 300, 100)
    Tw_n2 = np.linspace(Tw_min, Tw_max)

    np_new_L = np.linspace(500, 5000, 5)
    np_l_2 = [500, 750, 1250, 2000, 2500]
    L_best = max(prob2_solves)
    E_best = min(prob1_solves)
    E_worst = max(prob1_solves)
    for l_item in np_l_2:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        L_worst = l_item

        x = cvxpy.Variable(3, name='x')
        Fs = 1.0 / (15 * 60 * 1000)
        alpha_1, alpha_2, alpha_3 = calc_alphas(1)
        beta_1, beta_2 = calc_betas(D)
        Tt_x_3 = (x[2] / (Tps + Tal)) * ((Tps + Tal) / 2) + Tack + Tdata
        E1_tx_3 = (Tcs + Tal + Tt_x_3) * calc_f_out(1)
        obj_fun_game = - cvxpy.log(E_worst - x[0]) - cvxpy.log(L_worst - x[1])
        cons1 = E_worst
        cons2 = x[0]
        cons3 = L_worst
        cons4 = x[1]
        cons5 = x[2]
        cons6 = abs(calc_i_d(0)) * E1_tx_3
        constraints = [cons1 >= (alpha_1 * cvxpy.power(x[2], -1) + alpha_2 * x[2] + alpha_3),
                       cons2 >= (alpha_1 * cvxpy.power(x[2], -1) + alpha_2 * x[2] + alpha_3),
                       cons3 >= (beta_1 * x[2] + beta_2),
                       cons4 >= (beta_1 * x[2] + beta_2),
                       cons5 >= Tw_min,
                       cons6 <= (1 / 4)]

        prob3 = cvxpy.Problem(cvxpy.Minimize(obj_fun_game), constraints)
        try:
            result = prob3.solve()
        except cvxpy.SolverError:
            result = prob3.solve(solver=cvxpy.SCS)
        print("LMAX=" +str(l_item))
        print("optimal value p* = ", prob3.value)
        print("optimal var: E_1 = ", x[0].value)
        print("optimal var: L_1 = ", x[1].value)
        print("optimal var: T_w = ", x[2].value)
        #PLOTING WORST
        x_values = [E_worst, x[0].value]
        y_values = [L_worst, x[1].value]
        ax.scatter(E_worst, L_worst, color="green",
                   label="[Eworst, Lworst] = [" + str(E_worst) + ", " + str(L_worst) + "]")
        plt.plot(x_values, y_values, linestyle="--")

        #PLOTING BEST
        x_values = [E_worst, E_best]
        y_values = [L_worst, L_best]
        ax.scatter(E_best, L_best, color="orange", label="[Ebest, Lbest] = [" + str(E_best) + ", " + str(L_best) + "]")
        plt.plot(x_values, y_values, linestyle="--")

        ax.scatter(x[0].value, x[1].value, color="red",
                   label='Tradeoff Point with Lmax=' + str(l_item))

        plt.plot(energy_fun(Tw_n2), delay_fun(Tw_n2), color='b')
        plt.xlabel("E(Tw)")
        plt.ylabel("L(Tw)")
        plt.legend(loc="upper right")
        plt.savefig("game_theory_lmax_"+str(l_item)+".jpg")
        plt.show()

    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #E_BUDGET PLOT
    for e_item in np.linspace(0.05, 2.5, 50):
        E_worst = e_item
        L_worst = 500
        x = cvxpy.Variable(3, name='x')
        Tt_x_3 = (x[2] / (Tps + Tal)) * ((Tps + Tal) / 2) + Tack + Tdata
        E1_tx_3 = (Tcs + Tal + Tt_x_3) * calc_f_out(1)
        obj_fun_game = - cvxpy.log(E_worst - x[0]) - cvxpy.log(L_worst - x[1])
        cons1 = E_worst
        cons2 = x[0]
        cons3 = L_worst
        cons4 = x[1]
        cons5 = x[2]
        cons6 = abs(calc_i_d(0)) * E1_tx_3
        constraints = [cons1 >= (alpha1 * cvxpy.power(x[2], -1) + alpha2 * x[2] + alpha3),
                       cons2 >= (alpha1 * cvxpy.power(x[2], -1) + alpha2 * x[2] + alpha3),
                       cons3 >= (beta1 * x[2] + beta2),
                       cons4 >= (beta1 * x[2] + beta2),
                       cons5 >= Tw_min,
                       cons6 <= (1 / 4)]
    
        prob3 = cvxpy.Problem(cvxpy.Minimize(obj_fun_game), constraints)
        try:
            result = prob3.solve()
        except cvxpy.SolverError:
            result = prob3.solve(solver=cvxpy.SCS)
    
        print("optimal value p* = ", prob3.value)
        print("optimal var: E_1 = ", x[0].value)
        print("optimal var: L_1 = ", x[1].value)
        print("optimal var: T_w = ", x[2].value)
        if index == 1:  # FEASIBLE POINT
            ax.scatter(x[0].value, x[1].value, color=colours[colour_index % size_colours],
                       label='Tradeoff Point with Ebudget=' + str(round(e_item, 2)))
            colour_index += 1
        elif index % 10 == 0:
            ax.scatter(x[0].value, x[1].value, color=colours[colour_index % size_colours],
                       label='Tradeoff Point with Ebudget=' + str(round(e_item, 2)))
            colour_index += 1
        index += 1
    
    plt.plot(energy_fun(Tw_n), delay_fun(Tw_n), color='b')
    plt.xlabel("E(Tw)")
    plt.ylabel("L(Tw)")
    plt.legend(loc="upper right")
    plt.show()
    
    """