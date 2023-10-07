'''
Orbit diagram plotter - parallelized
Caleb Bessit
01 October 2023
'''

import math
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

#DECLARATION OF PARAMETERS

rmin            = 0
rmax            = 7*math.pow(10,4)
x_0             = 0.1       #Initial point  
y_0             = 0.2      
dr              = 1       #Parameter step 
trans           = 300       #Points taken as transient ("cool off") points
actual          = int(rmax)       #Number of points after transient to plot
k               = 8
mu              = 1.2
gain            = math.pow(10,k)
distribution    = np.ndarray.tolist( np.zeros(100))

#Generate r-values
R = np.arange(rmin, rmax, dr)


def f(x, y):
    global mu, gain
    # return r*x*(1-x)
    # return np.cos( r*np.arccos(x))
    # gain    = math.pow(10,r)
    a_star = np.cos( beta(y)*np.arccos(x) )
    b_star = np.cos( beta(x)*np.arccos(y) )
    return a_star*gain - np.floor(a_star*gain),  b_star*gain - np.floor(b_star*gain)

#Defines variable parameter for Chebyshev input
def beta(i):
    global mu
    return np.exp(mu*i*(1-i))

def calculateOrbit(r):
    global x_0, y_0, distribution
    x = x_0
    y = y_0
    # for i in range(trans):
    #     x, y = f(x, y)

    y_values = []
    x_values = []

    for i in range(actual):
        x, y = f(x, y)
        y_values.append(y)
        x_values.append(x)

        ind1, ind2 =  int(x*100), int(y*100)

        if ind1!=100:
            distribution[ind1]+=1
        else:
            distribution[99] +=1
        
        if ind2!=100:
            distribution[ind2]+=1
        else:
            distribution[99] +=1

    return y_values, x_values

def main():
    global R

    numCores = 4  

    resy, resx = calculateOrbit(0)

    # with Pool(numCores) as pool:
    #     results = pool.map(calculateOrbit, R)

    # y_values = []
    # x_values = []

    # for y in resy:
    #     y_values.extend(y)

    # for x in resx:
    #     x_values.extend(x)

    plt.scatter(resx, resy, color='cyan',s=0.5)
    plt.scatter(x_0, y_0, color= "red",label="(x(0), y(0))")
    plt.legend(loc="upper right")
    plt.xlabel("x_i")
    plt.ylabel("y_i")
    plt.title("Orbit diagram: fixed \u03BC=1.2, k=8")

    print(distribution)
    print(len(distribution))

    plt.figure()
    x_axis_vals = np.arange(0, 1, 0.01)
    plt.bar(x_axis_vals,distribution,width=0.008)
    # plt.xticks(np.arange(0,1.1,0.2))

    plt.title("Histogram of values for given intial condition")
    plt.xlabel("Value")
    plt.ylabel("Occurances")

    plt.show()

if __name__ == "__main__":
    main()

