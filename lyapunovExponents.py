#Lyapunov exponent plotter
#Caleb Bessit
#03 October 2023

import math
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from mpl_toolkits.mplot3d import Axes3D

#Create domain

x_0, y_0    = 0.1,0.2
points      = 35
mu_values   = np.linspace(0,10,points,endpoint=True)
k_values    = np.linspace(0,10,points,endpoint=True)
mu, K       = np.meshgrid(mu_values, k_values)
N           = 100 #Number of iterations as approximation of infinite sum

le1, le2 = 0,0

#Need to :
#1. Calculate Jacobian
#2. Calculate eigenvalues of Jacobian
#3. Calculate infinite sum
#4. Insert into an array
#5. Plot the surface as a function of mu and k

#Jacobian component functions
def dfdx(x_n,y_n, mu, k):
    return math.pow(10,k)*(   beta(y_n,mu)    *np.sin(beta(y_n,mu)*alpha(x_n))) / np.sqrt(1-x_n**2)

def dfdy(x_n,y_n, mu, k):
    return math.pow(10,k)*-1* mu* alpha(x_n)*(1-2*y_n)*beta(y_n, mu)*np.sin(beta(y_n,mu)*alpha(x_n))

def dgdx(x_n,y_n, mu, k):
    return math.pow(10,k)*-1* mu* alpha(y_n)*(1-2*x_n)*beta(x_n, mu)*np.sin(beta(x_n,mu)*alpha(y_n))

def dgdy(x_n,y_n, mu, k):
    return math.pow(10,k)*(   beta(x_n,mu)    *np.sin(beta(x_n,mu)*alpha(y_n))) / np.sqrt(1-y_n**2)

#Function components
def f(x,y, mu, k):
    a_star = np.cos( beta(y,mu)*np.arccos(x) )
    b_star = np.cos( beta(x,mu)*np.arccos(y) )
    gain = math.pow(10,k)
    return a_star*gain - np.floor(a_star*gain),  b_star*gain - np.floor(b_star*gain)


#Auxillary functions
def beta(i_n, mu):
    return np.exp(mu*i_n*(1-i_n))

def alpha(i_n):
    return np.arccos(i_n)

def calculateExponents(x_n, y_n, mu, k):
    global N
    sum1, sum2 = 0,0
    for n in range(1,N+1):
        x_n, y_n = f(x_n,y_n,mu, k)
        jacobian = np.array([[dfdx(x_n,y_n,mu, k), dfdy(x_n,y_n,mu, k)],
                             [dgdx(x_n,y_n,mu, k), dgdy(x_n,y_n,mu, k)]
                            ])
        
        lambda1, lambda2 = np.linalg.eigvals(jacobian)

        sum1 += np.log(abs(lambda1))
        sum2 += np.log(abs(lambda2))

    le1, le2 = (1/N)*sum1, (1/N)*sum2
    return le1, le2
def main():
    global mu, K, mu_values, k_values
    le1_values = []
    le2_values = []

    fig1 = plt.figure(figsize=(10,11))
    ax = plt.axes(projection='3d')

    for k in k_values:
        for m in mu_values:
            x_n= x_0
            y_n= y_0
            le1, le2 = calculateExponents(x_n, y_n,m, k)
            le1_values.append(le1)
            le2_values.append(le2)

    le1V = np.array(le1_values).reshape(len(mu_values),len(k_values))
    le2V = np.array(le2_values).reshape(len(mu_values),len(k_values))
    #Plotting figure
    
    surf1 = ax.plot_surface(mu,K, le1V, rstride=1, cstride=1, cmap='viridis_r', edgecolor='none' )

    ax.set_title("Lyapunov Exponent 1")
    ax.set_xlabel("\u03BC")
    ax.set_ylabel("k")
    ax.set_zlabel("\u03BB\u2081")
    
    #Adding colorbar
    colorbar = plt.colorbar(surf1, ax=ax, pad=0.05, shrink=0.5, aspect=10)
    colorbar.set_label('Values')

    #Plotting LE surface 2

    fig2 = plt.figure(figsize=(10,11))
    ax1 = plt.axes(projection='3d')
    surf2 = ax1.plot_surface(mu,K, le2V, rstride=1, cstride=1, cmap='viridis_r', edgecolor='none' )

    ax1.set_title("Lyapunov Exponent 2")
    ax1.set_xlabel("\u03BC")
    ax1.set_ylabel("k")
    ax1.set_zlabel("\u03BB\u2082")
    
    #Adding colorbar
    colorbar = plt.colorbar(surf2, ax=ax1, pad=0.05, shrink=0.5, aspect=10)
    colorbar.set_label('Values')
    plt.show()


    
if __name__=="__main__":
    main()