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

def betas(d):
    beta1 = sum([1/2]*d)
    beta2 = sum([Tcw / 2 + Tdata]*d)

    return beta1, beta2

def energy_fun(tw):  ##ENERGY FUNCTION
    return alpha1/tw + alpha2*tw + alpha3

def delay_fun(tw):
    return beta1*tw + beta2

if __name__ == "__main__":

    time = [1, 5, 10, 15, 20, 25]

    for t in time:

        Fs = 1.0/(t * 60 * 1000)
        alpha1, alpha2, alpha3 = alphas(1)
        beta1, beta2 = betas(D)
        #x = Variable(3, name='x')

        # Problem 1
        # obj_fun = alpha1/Tw + alpha2 * Tw + alpha3#E(Tw) -> energy
        # cons1 = beta1 * Tw + beta2
        # cons2 =
        # cons3 =
        # cons4 =
        # constraints = [cons1 <= Lmax, cons2 >= Tw_min, cons3 <= Tw_max, cons4 <= (1/4) ]

        x = np.linspace(Tw_min, Tw_max)

        plt.plot(x, energy_fun(x), color='blue', label='fun')
        plt.xlabel('Tw')
        plt.ylabel('Energy consumed')
        plt.title('Energy function of Tw with Fs = ' + str(t) + " pkt/min")
        plt.show()

        plt.plot(x, delay_fun(x), color='blue', label='fun')
        plt.xlabel('Tw')
        plt.ylabel('Delay time')
        plt.title('Delay function of Tw with Fs = ' + str(t) + " pkt/min")
        plt.show()

        plt.plot(energy_fun(x), delay_fun(x), color='blue', label='fun')
        plt.xlabel('Energy')
        plt.ylabel('Delay')
        plt.title('Energy-Delay function with Fs = ' + str(t) + " pkt/min")
        plt.show()

