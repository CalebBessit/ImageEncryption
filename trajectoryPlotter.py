'''
Trajectory analyzer for iterated maps
Caleb Bessit
01 October 2023
'''

import numpy as np
import math
import matplotlib.pyplot as plt

r       = 3.95
mu      = r
x_0     = 0.1
y_0     = 0.1
y_0P    = 0.1 + math.pow(10,-15)
x_0P    = 0.1 + math.pow(10,-15)
k       = 3
gain    = math.pow(10,k)
its     = 100

def f(x,y):
    global r, mu, gain
    # return r*x*(1-x)
    #return np.cos(r*np.arccos(x))

    a_star = np.cos( beta(y)*np.arccos(x) )
    b_star = np.cos( beta(x)*np.arccos(y) )
    return a_star*gain - np.floor(a_star*gain),  b_star*gain - np.floor(b_star*gain)

#Defines variable parameter for Chebyshev input
def beta(i):
    global mu
    return np.exp(mu*i*(1-i))

def main():
    global x_0, x_0P, its, y_0, y_0P
    x_valsBase      = [x_0]
    x_valsPertubed  = [x_0P]
    iterations      = [1]

    for i in range(2,its+2):
        x_0, y_0 = f(x_0,y_0)
        x_0P, y_0P = f(x_0P, y_0P)

        x_valsBase.append(x_0)
        x_valsPertubed.append(x_0P)

        iterations.append(i)

    plt.xlabel("Iterations")
    plt.ylabel("Value of x")
    plt.title("100 iterations of 2D-LCCM")

    plt.plot(iterations, x_valsBase, color="red", label="x=0.1")
    plt.plot(iterations, x_valsPertubed, color="blue", label="x=0.1+10^-15")
    plt.legend(loc='upper right')

    plt.show()

if __name__ == "__main__":
    main()
