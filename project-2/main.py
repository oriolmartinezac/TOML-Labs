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
    time = [1, 5, 10, 15, 20, 25]
    alpha_1, alpha_2, alpha_3 = 0.0, 0.0, 0.0
    x = np.linspace(Tw_min, Tw_max)

    for t in time:
        Fs = 1.0 / (t * 60 * 1000)
        alpha_1, alpha_2, alpha_3 = calc_alphas(1)
        label = "" + str(round(1 / t, 3)) + " pkt/min"
        plt.plot(x, energy_fun(x), label=label)
        plt.xlabel('Tw')
        plt.ylabel('Energy consumed')
        plt.title("Energy function of Tw with different Fs")

    plt.legend()
    plt.show()

    beta_1, beta_2 = calc_betas(D)
    plt.plot(x, delay_fun(x), color='red', label='fun')
    plt.xlabel('Tw')
    plt.ylabel('Delay time')
    plt.title("Delay function of Tw")
    plt.show()

    for t in time:
        Fs = 1.0 / (t * 60 * 1000)
        alpha_1, alpha_2, alpha_3 = calc_alphas(1)
        beta_1, beta_2 = calc_betas(D)
        title = "" + str(round(1 / t, 3)) + " pkt/min"
        plt.plot(energy_fun(x), delay_fun(x), color="green")
        plt.xlabel('Energy')
        plt.ylabel('Delay')
        plt.title("Energy-Delay function with " + title)
        plt.show()

# PART 2 #
x = gpkit.Variable("x")

alpha1, alpha2, alpha3 = calc_alphas(1)
beta1, beta2 = calc_betas(D)
Tt_x = (x / (Tps + Tal)) * ((Tps + Tal) / 2) + Tack + Tdata
E1_tx = (Tcs + Tal + Tt_x) * calc_f_out(1)
prob1_solves = []
np_L = np.linspace(100, 5000)

for l_item in np_L:
    obj_fun1 = alpha1 / x + alpha2 * x + alpha3
    cons1 = beta1 * x + beta2
    cons2 = x
    cons3 = abs(calc_i_d(0)) * E1_tx
    constraints = [cons1 <= l_item, cons2 >= Tw_min, cons3 <= (1 / 4)]
    prob1 = gpkit.Model(obj_fun1, constraints)
    solution = prob1.solve()
    print(solution['cost'])
    prob1_solves.append(solution["cost"])

plt.plot(np_L, prob1_solves, color="blue")
plt.xlabel('L_max')
plt.ylabel('Minimization')
plt.title("Problem 1 to optimize")
plt.show()

prob2_solves = []
np_E = np.linspace(0.5, 5)

for e_item in np_E:
    obj_fun2 = beta1 * x + beta2
    cons1 = alpha1 / x + alpha2 * x + alpha3
    cons2 = x
    cons3 = abs(calc_i_d(0)) * E1_tx
    constraints = [cons1 <= e_item, cons2 >= Tw_min, cons3 <= (1 / 4)]
    prob2 = gpkit.Model(obj_fun2, constraints)
    solution = prob2.solve()
    print(solution['cost'])
    prob2_solves.append(solution["cost"])

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

# PART 3 #
# Game
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
alpha1, alpha2, alpha3 = calc_alphas(1)
beta1, beta2 = calc_betas(D)

colours = ['red', 'green', 'yellow', 'black', 'purple', 'orange']
size_colours = len(colours)
Tw_n = np.linspace(50, 300, 100)
index = 1
colour_index = 0
for l_item in np_L:
    L_worst = l_item
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
    if index == 5:  # FEASIBLE POINT
        ax.scatter(x[0].value, x[1].value, color=colours[colour_index % size_colours],
                   label='Tradeoff Point with Lmax=' + str(l_item))
        colour_index += 1
    elif index % 10 == 0:
        ax.scatter(x[0].value, x[1].value, color=colours[colour_index % size_colours],
                   label='Tradeoff Point with Lmax=' + str(l_item))
        colour_index += 1
    index += 1

plt.plot(energy_fun(Tw_n), delay_fun(Tw_n), color='b')
plt.xlabel("E(Tw)")
plt.ylabel("L(Tw)")
plt.legend(loc="upper right")
plt.show()
