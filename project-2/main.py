from header import *  ##IMPORTING HEADER FILE
import matplotlib.pyplot as plt
from cvxpy import *

def N_d(d):
    n_d = (2*d - 1)*C
    if d == 0:
        n_d = 1
    return n_d

def I_d(d):
    if d == 0:
        i_d = C
    elif d == D:
        i_d = 0
    else:
        i_d = (2 * d + 1) / (2 * d - 1)
    return i_d

def F_out(d):
    f_out = Fs * ((D**2 - d**2 + 2*d - 1) / (2*d - 1))
    if d == D:
        f_out = Fs
    return f_out

def F_B(d):
    return C - I_d(d) * F_out(d)

def F_I(d):
    f_i = Fs*((D**2 - d**2) / (2*d - 1))
    if d == 0:
        f_i = Fs*(D**2)*C
    return f_i

def alphas(d):
    alpha1 = Tcs + Tal + 3 / 2 * Tps * ((Tps + Tal) / 2 + Tack + Tdata) * F_B(d)
    alpha2 = F_out(d)/2
    alpha3 = ((Tps + Tal) / 2 + Tcs + Tal + Tack + Tdata) * F_out(d) + (
                3 / 2 * Tps + Tack + Tdata) * F_I(d) + 3 / 4 * Tps * F_B(d)

    return alpha1, alpha2, alpha3

def energy_fun(tw):  ##ENERGY FUNCTION
    return alpha1/tw + alpha2*tw + alpha3

def delay_fun(tw):
    return

if __name__ == "__main__":

    d = 1
    alpha1, alpha2, alpha3 = alphas(d)

    beta1 = sum(1 / 2)
    beta2 = sum(Tcw / 2 + Tdata)




    #x = Variable(3, name='x')

    # Problem 1
    # obj_fun = alpha1/Tw + alpha2 * Tw + alpha3#E(Tw) -> energy
    # cons1 = beta1 * Tw + beta2
    # cons2 =
    # cons3 =
    # cons4 =
    # constraints = [cons1 <= Lmax, cons2 >= Tw_min, cons3 <= Tw_max, cons4 <= (1/4) ]

    x_dummy1 = np.linspace(Tw_min, Tw_max)

    plt.plot(x_dummy1, energy_fun(x_dummy1), color='blue', label='fun')
    plt.show()
