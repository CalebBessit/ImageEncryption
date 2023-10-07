'''
Orbit diagram plotter
Caleb Bessit
01 October 2023
'''

import numpy as np
import matplotlib.pyplot as plt

#DECLARATION OF PARAMETERS

rmin   = 0.9
rmax   = 2.5
x_0     = 0.5       #Initial point        
dr      = 0.01       #Parameter step 
trans   = 300       #Points taken as transient ("cool off") points
actual  = 300       #Number of points after transient to plot

#Generate r-values
R = np.arange(rmin, rmax, dr)


def f(x, r):
    # return r*x*(1-x)
    return np.cos( r*np.arccos(x))
def main():
    #Iterate over r-values and plot
    global R, x_0
    x = x_0
    for r in R:
        for i in range(trans):
            x = f(x,r)


        for i in range(actual):
            x = f(x,r)
            plt.scatter(r,x,color='black',s=1)
    

    plt.xlabel("Parameter: r")
    plt.ylabel("x")
    plt.title("Orbit diagram")
    plt.show() 

if __name__=="__main__":
    main()


