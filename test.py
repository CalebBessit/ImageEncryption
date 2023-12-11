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
#     xvals = [0.4841814169096895, 0.9474353332015413, 0.7395971017211852, 0.9999374981720855, 0.22323222277239874, 0.018155628036470417, 0.8324946547379329, 0.651144528442126, 0.7145492592050556, 0.8084115184485581, 0.5760996820607261, 0.832155516497767, 0.5946186560223359, 0.11301439632285426, 0.29618148216900175, 0.9396643501560743, 0.42872606796134183, 0.7766966637646269, 0.16416550923182527, 0.17256919088560962]
#     yvals = [0.9883516920516855, 0.8389044354779558, 0.6617234191677495, 0.783178935567716, 0.15289969379040858, 0.2170232575381904, 0.24986774533106504, 0.14104828004238834, 0.49778228315163486, 0.6670554231664199, 0.8842208326968592, 0.860766951239447, 0.09926534526862563, 0.4018547632153111, 0.9151952006948302, 0.4537001534105267, 0.901333803354937, 0.8172031373847968, 0.14699566429911393, 0.32546679047428895]
#     global n
#     # xvals, yvals =[],[]

#     # for k in range(n):
#     #     xvals.append(random.random())
#     #     yvals.append(random.random())
    
#     values = brownianMotion(x_n, y_n,z_n, xvals, yvals)
#     X, Y, Z = [],[],[]
#     for v in values:
#         X.append(v[0])
#         Y.append(v[1])
#         Z.append(v[2])

#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     ax.plot3D(X,Y,Z)
#     ax.scatter(x_0,y_0,z_0,c="red",label="Starting point")
#     ax.set_title("20 steps of Brownian Motion")
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_zlabel("z")
#     ax.legend()
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

'''8-bit binary test'''
# integer_value = 230  # Replace this with your integer value
# binary_value = bin(integer_value & 0xFF)[2:].zfill(8)

# print(binary_value, binary_value[0:4], binary_value[4:])  # Output: '00101010'

'''Method within method test'''
# def main():
#     a= 2
#     b = 3

#     def job():
#         return a+b
    
#     print(job())

# main()

'''Binary test'''
# k = 6
# m = bin(6)
# j = int(m,2)

# print(bin())

'''Bitwise test'''
# import math
# import numpy as np
# def generateChaoticMatrices(x2n, y2n,K):
#     a1, a2   = [], []
#     gain = math.pow(10,8)
#     for a in range(K*K):
#          #Calculate the values for the chaotic matrices and convert them to 8-bit binary values
#          t1 = np.mod( int( (  x2n[a]    + y2n[a]    +1 ) * gain ), 16)
#          a1.append( bin(t1 & 0xFF)[2:].zfill(8) )

#          t1 = np.mod( int( (  x2n[a]**2 + y2n[a]**2 +1 ) * gain ), 16) 
#          a2.append( bin(t1 & 0xFF)[2:].zfill(8)  )

#     a1 = np.array(a1).reshape(K,K)
#     a2 = np.array(a2).reshape(K,K)

#     return a1, a2

# def main():
#      K = 3
#      xvals = [0.3915593987763005, 0.48328774294384524, 0.576855968516515, 0.6539126774036531, 0.5236793746667767, 0.3840997701028661, 0.9161186287674689, 0.18053618216359002, 0.4999903938518415, 0.48848676027828164, 0.8002361335792902, 0.8351344683453209, 0.1811578417199705, 0.9244854775284312, 0.5832192729403348, 0.9950146558625067, 0.19158578228628753, 0.5164149311582781, 0.24912404591617143, 0.408793584004603, 0.7218831091398992, 0.6468693631490148, 0.5853346521360683, 0.6545767189954402, 0.4980124338306018, 0.32134026034608165, 0.5526599895503314, 0.5434534983744351, 0.4773209170800218, 0.9287124834420343, 0.6253505935773253, 0.20178758209041436, 0.5499209415995651, 0.8212856160774566, 0.6468390965322285, 0.44909275801373916, 0.2982369955181309, 0.26729156770286333, 0.41197254098130354, 0.8027264132760302, 0.10008701948507626, 0.4075844417209955, 0.1902471950896215, 0.39156444637302945, 0.6952464599357452, 0.17710940788611018, 0.12740406320208553, 0.1978938627572926, 0.36046965762646366, 0.4816581832546105, 0.2772041751709413, 0.6017937534699108, 0.4109367796499004, 0.6791582306009284, 0.5313242982087943, 0.15891075615971828, 0.987032309089276, 0.1748321120764268, 0.3494156374839894, 0.18005664428190082, 0.6243692318460634, 0.8838846574766245, 0.8909522557850756, 0.14425494189991217, 0.8517444043693371, 0.9281219698976285, 0.2718192356375866, 0.968910732612827, 0.08523814522590911, 0.06915985529420732, 0.5862243928042473, 0.7797294320525535, 0.16410423598554758, 0.9322836223906793, 0.033540938788405805, 0.6899787281757603, 0.9325276743210225, 0.4610811135936512, 0.1851373438729298, 0.9835275484875295, 0.09218520838644806, 0.9756344534152238, 0.34985255731334874, 0.8453436629090479, 0.3805195681046031, 0.7894816970536563, 0.10625191495010722, 0.9965072203837287, 0.04770434833959758, 0.006267042253650379, 0.20671506055958977, 0.7622226850128133, 0.28832219102230106, 0.33918789584436226, 0.8968903100235655, 0.8204646913978864, 0.4093338961090991, 0.7026164312935166, 0.8211384333831984, 0.5714676943509384]
#      yvals = [0.31999314650447275, 0.4577332794749851, 0.9144638542739636, 0.043826708233147826, 0.9963821618277103, 0.5409440493187813, 0.08025416521247608, 0.6321240930996064, 0.8431398297968066, 0.7789399266415411, 0.01762578070495946, 0.9353236484571728, 0.08504156860772394, 0.4318401835022575, 0.9245390274334663, 0.6391096251893159, 0.013430243688601462, 0.6722558847133402, 0.6511954011735867, 0.5337448720384655, 0.5007417395068651, 0.7408374791318101, 0.8524408420672549, 0.8009512926667315, 0.567821754325265, 0.7340991517470377, 0.417518747086275, 0.9884918847068733, 0.8473398592851094, 0.017861031405393746, 0.9369232471945113, 0.2432571665536497, 0.6345723033373629, 0.7936454339192918, 0.8950880827908466, 0.7060319915470629, 0.1213463209553014, 0.9474252480708125, 0.4102348630867638, 0.14745957300985058, 0.4431244304705332, 0.5450389758079567, 0.7506171263009606, 0.9120058513230394, 0.3012473933516284, 0.3441579924389484, 0.9457177435404924, 0.8744041848985904, 0.7179760574760398, 0.04645285642559294, 0.15453356317357592, 0.5232335442010105, 0.8915419817450883, 0.005327109867091062, 0.733696747969535, 0.26439602519091066, 0.8211321694540089, 0.32574147063750025, 0.8175734520855904, 0.21164369845551945, 0.7054278638145156, 0.08964550823797379, 0.11407844814778545, 0.5622050841493739, 0.9023156253609063, 0.4183032202173662, 0.9721060245873601, 0.3114824692745599, 0.882681889200972, 0.40732910958140733, 0.6517723351543968, 0.4548977014607569, 0.5838666193885526, 0.3608716958111168, 0.8675690027675432, 0.7509592294566251, 0.4508924893271299, 0.809684800165021, 0.3434233353655629, 0.31152717403822017, 0.4495551024205424, 0.5955158654974566, 0.20352577027296237, 0.9416444364789537, 0.9461762121847076, 0.3121186146236955, 0.2431140645582156, 0.9107358448329553, 0.06861754644516116, 0.44706062920780754, 0.6015038636104556, 0.5141250154200089, 0.5134424691911678, 0.67605730268757, 0.5415802835222229, 0.3077383955688252, 0.6006265309664846, 0.13839218486237448, 0.7092363177354294, 0.19321948506940712]
   
#      a1, a2 = generateChaoticMatrices(xvals[0:9],yvals[0:9],K)
#      print(a1)
#      print(a2)

# main()

'''Bitwise float test'''

# import struct

# # Step 1: Convert a float to its binary representation
# def float_to_bin(num):
#     # Use 'd' format for double precision (64-bit) floating-point number
#     binary_representation = struct.pack('d', num)
#     binary_string = ''.join(f'{byte:08b}' for byte in binary_representation)
#     return binary_string

# # Step 2: Perform bitwise XOR operation on two binary representations
# def perform_bitwise_xor(float1, float2):
#     binary_str1 = float_to_bin(float1)
#     binary_str2 = float_to_bin(float2)

#     # Perform XOR operation bit by bit
#     xor_result = ''.join(['1' if bit1 != bit2 else '0' for bit1, bit2 in zip(binary_str1, binary_str2)])

#     return xor_result

# # Step 3: Convert the XOR result back to a float
# def bin_to_float(binary_string):
#     bytes_list = [int(binary_string[i:i+8], 2) for i in range(0, len(binary_string), 8)]
#     packed_bytes = bytes(bytearray(bytes_list))
#     unpacked_float = struct.unpack('d', packed_bytes)[0]
#     return unpacked_float

# # Example usage:
# float1 = 3.14159
# float2 = 2.71828

# xor_result = perform_bitwise_xor(float1, float2)
# resulting_float = bin_to_float(xor_result)
# test = bin_to_float(perform_bitwise_xor(float2,resulting_float))

# print(f'Float 1: {float1}')
# print(f'Float 2: {float2}')
# print(f'XOR result binary: {xor_result}')
# print(f'Resulting float: {resulting_float}')
# print("This should be pi: {}".format(test))

# import struct
# import numpy as np

# def float_to_bin(num):
#     binary_representation = struct.pack('d', num)
#     binary_string = ''.join(f'{byte:08b}' for byte in binary_representation)
#     return np.array([int(bit) for bit in binary_string])

# def perform_bitwise_xor(float1, float2):
#     binary_str1 = float_to_bin(float1)
#     binary_str2 = float_to_bin(float2)
#     xor_result = np.bitwise_xor(binary_str1, binary_str2)
#     return ''.join(map(str, xor_result))

# def bin_to_float(binary_string):
#     bytes_list = [int(binary_string[i:i+8], 2) for i in range(0, len(binary_string), 8)]
#     packed_bytes = bytes(bytearray(bytes_list))
#     unpacked_float = struct.unpack('d', packed_bytes)[0]
#     return unpacked_float

# # Example usage:
# float1 = 3.14159
# float2 = 2.71828

# xor_result = perform_bitwise_xor(float1, float2)
# resulting_float = bin_to_float(xor_result)
# test = bin_to_float(perform_bitwise_xor(float2,resulting_float))

# print(f'Float 1: {float1}')
# print(f'Float 2: {float2}')
# print(f'XOR result binary: {xor_result}')
# print(type(xor_result))
# print(f'Resulting float: {resulting_float}')
# print("This should be pi: {}".format(test))


'''Swapping rows and columns test'''

# import numpy as np

# def exchange_rows(arr1, arr2, index):
#     # Ensure both arrays have the same shape
#     if arr1.shape != arr2.shape:
#         raise ValueError("Arrays must have the same shape")

#     # Swap rows between arrays
#     arr1[index], arr2[index] = arr2[index].copy(), arr1[index].copy()

#     return arr1, arr2

# # Test arrays
# A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# B = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])

# # Call the function to exchange rows based on index 1
# result_A, result_B = exchange_rows(A.copy(), B.copy(), 1)

# print("Array A after exchanging rows:")
# print(result_A)
# print("\nArray B after exchanging rows:")
# print(result_B)

'''Writing/reading arrays to file test'''
import numpy as np

# Assuming you have numpy arrays arr1, arr2, arr3
arr1 = np.array([[1, 2, 3],[4,5,6],[7,8,9]])
arr2 = np.array([[4, 5,7],[1,2,3],[5,6,7]])
arr3 = np.array([[8, 9, 10],[11,12,12],[13,12,12]])

# Save the arrays to a file
np.save('test.npy', [arr1, arr2, arr3])

import numpy as np

# Load the arrays from the file
loaded_arrays = np.load('test.npy', allow_pickle=True)

# Retrieve the arrays from the loaded file
loaded_arr1,loaded_arr2,loaded_arr3 = loaded_arrays

print(loaded_arrays)
print()
# Now you can use loaded_arr1, loaded_arr2, loaded_arr3 in your
print(loaded_arr1)
print()
print(loaded_arr2)
print()
print(loaded_arr3)

# line = "Hash code: dfd89ba48a86717f3617685a7018a0f6ed98ce84c39a2171c1418fdc90769dff"
# file = open("test.txt","w")
# print(line,file=file, sep="")
# file.close()

# file = open("test.txt","r")
# line = file.readline()
# print(line[line.rfind(":")+2:])
# file.close()


