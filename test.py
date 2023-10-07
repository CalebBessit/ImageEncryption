# # import matplotlib.pyplot as plt
# # import numpy as np

# # # Your array of values (example values)
# # values = np.random.rand(100)  # Replace with your actual values

# # # Generate the corresponding x-axis values (0 to 1 in 0.01 increments)
# # x_values = np.arange(0, 1, 0.01)

# # # Create the bar chart
# # plt.bar(x_values, values, width=0.01)  # Width is set to 0.01 to match bin width

# # # Set the x-axis ticks and labels in multiples of 0.2
# # x_ticks = np.arange(0, 1.1, 0.2)
# # plt.xticks(x_ticks)

# # # Label the axes
# # plt.xlabel('X-axis Label')
# # plt.ylabel('Y-axis Label')

# # # Show the plot
# # plt.show()


# # import numpy as np

# # # Define a 2x2 matrix (example)
# # matrix = np.array([[2, 0],
# #                    [0, 3]])

# # # Calculate the eigenvalues
# # # eigenvalues = np.linalg.eigvals(matrix)
# # lambda1, lambda2 = np.linalg.eigvals(matrix)
# # print("Eigenvalues:", lambda1, lambda2)


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

# if __name__ == "__main__":
#     main()
import numpy as np
m = [1,2,3,4,5,6,7,8,9]
k = np.array(m)
j = k.reshape(3,3)
print(j)
row = 2
col = 1
print(j[row-1][col-1])

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
