#Cobweb diagram plotter
#Caleb Bessit
#29 June 2023

import numpy as np
import matplotlib.pyplot as plt
import sys

#Global variable
ppl  = 100     #Points per unit length for plot (resolution)
tol  = 10e-5   #Tolerance of convergence
runs = 200     #Default number of iterations
a    = 0.5     #Initial guess
b    = 0       #Intial point of cobweb diagram is on x-axis
r    = 3.9

def f(x):
     # return np.cos(r*np.arccos(x))
     return r*x*(1-x)

def main():
     global a
     global b
     global runs
     global ppl
     global tol

     values =[]  #Array of all the values generated. Used for scaling plot accordingly

     i = 0
     diff = sys.maxsize  #To ensure iterations terminate if suitable convergence is found
     while (i<runs) and (diff>tol):
          i+=1

          #Plot the point (a,b), update b, plot the vertical line from (a,y1) to (a, y2)
          if i==1:
               plt.text(a,b, " x\u2080={}".format(a))


          plt.scatter(a,b, color="black")
          y1 = b
          b = f(a)
          values.append(a)
          values.append(b)

          verticalLine(a, y1, b)

          #Plot the point (a,b) again, update a, plot the horizontal line from (x1,a) to (x2, a)
          plt.scatter(a,b, color="black")
          x1 = a
          a = b

          values.append(a)

          horizontalLine(x1, a, b)

          diff = abs(a - f(a))

     #Graph configuration. Plot y=x and y=f(x)

     width = max(values)-min(values)
     points = int(round(width*ppl))
     # X = np.linspace(min(values)-0.1*width,max(values)+0.1*width, points)
     X = np.linspace(0,1,points)
     Y = f(X)

     plt.plot(X,X, label="y=x")
     plt.plot(X,Y, label="y=f(x)")

     plt.xlabel("x= x\u2099")
     plt.ylabel("y= x \u2099\u207A\u2081")
     plt.title("Cobweb diagram for y=f(x)")
     plt.legend()
     plt.show()


def horizontalLine(x1, x2, y):
     global ppl

     width = abs(x2 - x1)
     points = int(round(width*ppl))

     a = min(x1, x2)
     b = max(x1, x2)

     X = np.linspace(a, b, points)
     Y = np.full(points, y)
     plt.plot(X, Y, color="black")

def verticalLine(x, y1, y2):
     global ppl

     width = abs(y2 - y1)
     points = int(round(width*ppl))

     a = min(y1, y2)
     b = max(y1, y2)

     X = np.full(points, x)
     Y = np.linspace(a, b, points)
     plt.plot(X, Y, color="black")

if __name__=="__main__":
     main()
