'''Eigenvalues test'''
# # import numpy as np

# # # Define a 2x2 matrix (example)
# # matrix = np.array([[2, 0],
# #                    [0, 3]])

# # # Calculate the eigenvalues
# # # eigenvalues = np.linalg.eigvals(matrix)
# # lambda1, lambda2 = np.linalg.eigvals(matrix)
# # print("Eigenvalues:", lambda1, lambda2)

'''Orbit diagram temporary'''
# '''
# Orbit diagram plotter - parallelized
# Caleb Bessit
# 01 October 2023
# '''

# import numpy as np
# import matplotlib.pyplot as plt
# from multiprocessing import Pool

# #DECLARATION OF PARAMETERS

# # rmin    = 3.83
# # rmax    = 3.855
# import numpy as np
# import math
# import matplotlib.pyplot as plt

# r       = 3.95
# mu      = r
# x_0     = 0.1
# y_0     = 0.1
# y_0P    = 0.1 + math.pow(10,-15)
# x_0P    = 0.1 + math.pow(10,-15)
# k       = 3
# gain    = math.pow(10,k)
# its     = 100

# def f(x,y):
#     global r, mu, gain
#     return r*x*(1-x),0
#     # return np.cos(r*np.arccos(x)),0

#     # a_star = np.cos( beta(y)*np.arccos(x) )
#     # b_star = np.cos( beta(x)*np.arccos(y) )
#     # return a_star*gain - np.floor(a_star*gain),  b_star*gain - np.floor(b_star*gain)

# #Defines variable parameter for Chebyshev input
# def beta(i):
#     global mu
#     return np.exp(mu*i*(1-i))

# def main():
#     global x_0, x_0P, its, y_0, y_0P
#     x_valsBase      = [x_0]
#     x_valsPertubed  = [x_0P]
#     iterations      = [1]

#     for i in range(2,its+2):
#         x_0, y_0 = f(x_0,y_0)
#         x_0P, y_0P = f(x_0P, y_0P)

#         x_valsBase.append(x_0)
#         x_valsPertubed.append(x_0P)

#         iterations.append(i)

#     plt.xlabel("Iterations")
#     plt.ylabel("Value of x")
#     plt.title("100 iterations of Logistic Map")

#     plt.plot(iterations, x_valsBase, color="red", label="x=0.1")
#     plt.plot(iterations, x_valsPertubed, color="blue", label="x=0.1+10^-15")
#     plt.legend(loc='upper right')

#     plt.show()


'''Array and reshaping test'''
# if __name__ == "__main__":
#     main()
# import numpy as np
# m = [1,2,3,4,5,6,7,8,9]
# k = np.array(m)
# j = k.reshape(3,3)
# print(j)
# row = 2
# col = 1
# print(j[row-1][col-1])

'''SHA256 and bitwise operations test'''
# import hashlib

# # Create a hashlib object for SHA-256
# sha256_hash = hashlib.sha256()

# # Encode the string to bytes and update the hash
# data_stream = "Hello world!".encode('utf-8')
# sha256_hash.update(data_stream)

# # Get the hexadecimal representation of the hash
# hex_digest = sha256_hash.hexdigest()

# key1 = hex_digest[0:32]
# key2 = hex_digest[32:]

# key3 = hex(int(key1, 16) ^ int(key2, 16))
# # print(key1, key2, "\nKey 3: ", key3)
# strKey3 = str(key3)[2:]
# print(strKey3[0:8])

# epsilonValues=[]

# count = 0
# hVals = []
# runningH = 0
# for h in strKey3:
#     count+=1

#     if count==1:
#         runningH = int(h,16)
#     else:
#         runningH = runningH ^ int(h,16)

#     if count==8:
#         count=0
#         runningH /= 15
#         epsilonValues.append(runningH)

# print(epsilonValues)



# Print the result
# print(hex_digest)
# print(len(hex_digest), len(key1), len(key2), len(strKey3))

'''Brownian motion test'''
# import math
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d

# r = 100
# n = 20

# def rho(x_n,y_n):
#     return np.mod(  np.floor(  (x_n+y_n)*math.pow(10,8)) ,r+1   )   

# #Written in the paper as theta_1
# def phi(x_n):
#     return np.deg2rad(  np.mod( np.floor(x_n * math.pow(10,8)) ,181  )  )

# #Written in the paper as theta_2
# def theta(y_n):
#     return np.deg2rad(  np.mod(  np.floor(y_n* math.pow(10,8)), 361  )  )



# def x(rho, phi, theta):
#     return rho*np.sin(phi)*np.cos(theta)

# def y(rho, phi, theta):
#     return rho*np.sin(phi)*np.sin(theta)

# def z(rho, phi):
#     return rho*np.cos(phi)

# def brownianMotion(x_n,y_n,z_n,xStream,yStream):
#     global n, r

#     values = [ ((x_n,y_n,z_n))  ]
#     for m in range(n):
#         r_update        = rho(xStream[m],yStream[m])
#         theta_1_update  = phi(xStream[m])
#         theta_2_update  = theta(yStream[m])

#         x_n = x_n + x(r_update, theta_1_update, theta_2_update)
#         y_n = y_n + y(r_update, theta_1_update, theta_2_update)
#         z_n = z_n + z(r_update, theta_1_update)

#         values.append(  (x_n, y_n,z_n))

#     return values


# def main():
#     K = 100
#     x_n, y_n, z_n=  0,0,0

#     x_0, y_0, z_0 = x_n, y_n, z_n
#     # xvals = [0.4841814169096895, 0.9474353332015413, 0.7395971017211852, 0.9999374981720855, 0.22323222277239874, 0.018155628036470417, 0.8324946547379329, 0.651144528442126, 0.7145492592050556, 0.8084115184485581, 0.5760996820607261, 0.832155516497767, 0.5946186560223359, 0.11301439632285426, 0.29618148216900175, 0.9396643501560743, 0.42872606796134183, 0.7766966637646269, 0.16416550923182527, 0.17256919088560962]
#     # yvals = [0.9883516920516855, 0.8389044354779558, 0.6617234191677495, 0.783178935567716, 0.15289969379040858, 0.2170232575381904, 0.24986774533106504, 0.14104828004238834, 0.49778228315163486, 0.6670554231664199, 0.8842208326968592, 0.860766951239447, 0.09926534526862563, 0.4018547632153111, 0.9151952006948302, 0.4537001534105267, 0.901333803354937, 0.8172031373847968, 0.14699566429911393, 0.32546679047428895]
#     global n
#     xvals, yvals =[],[]

#     for k in range(n):
#         xvals.append(random.random())
#         yvals.append(random.random())
    
#     values = brownianMotion(x_n, y_n,z_n, xvals, yvals)
#     X, Y, Z = [],[],[]
#     for v in values:
#         X.append(v[0])
#         Y.append(v[1])
#         Z.append(v[2])

#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     ax.plot3D(X,Y,Z)
#     ax.scatter(x_0,y_0,z_0,c="red")
#     plt.show()

#     xNorm = []
#     maxVal = max(X)
#     minVal = min(X)
#     for m in X:
#         xNorm.append(   ((m-minVal)*K)/(maxVal-minVal)    )

#     print(X)
#     print(xNorm)

# main()


'''Ranking array test'''

# import numpy as np

# array = np.array([1,28,14,25,9,3])

# temp = array.argsort()
# print(temp)
# ranks = np.empty_like(temp)
# print(ranks)
# ranks[temp] = np.arange(len(array))
# print(array)
# print(ranks)


'''Print test'''
# K = 10
# fileHeader = "P2\n# Scrambled Image\n{} {}\n255\n".format(K,K)
    
# Q_2 = [1,2,3,4,5,6,7,8]

# for f in range(len(Q_2)):
#         Q_2[f] = str(Q_2[f]) + "\n"

# fileContent = "".join(Q_2)
# fileContent = fileHeader + fileContent
# print(fileContent)1

'''Inverse ranking test'''
# import numpy as np

# m = [4, 2, 0, 1, 3]

# new = np.zeros(len(m)).tolist()

# count = -1
# for l in m:
#     count+=1
#     new[l] = count

# print(new)

# import numpy as np

# m = [4, 2, 0, 1, 3]

# p = np.asanyarray(m)
# s = np.empty_like(p)
# s[p] = np.arange(p.size)

# s = s.tolist()
# print(s)
