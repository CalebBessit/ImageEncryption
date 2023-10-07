#Image Encryption using a the 2D-LCCM
#Caleb Bessit
#05 October 2023

import math
import hashlib
import numpy as np

#INITIAL KEYS
x_0, y_0, mu, k = 1,1,1,1
gain            = math.pow(10,k)
n               = 20
r               = 100

def f(x, y):
    global mu, gain
    
    a_star = np.cos( beta(y)*np.arccos(x) )
    b_star = np.cos( beta(x)*np.arccos(y) )
    return a_star*gain - np.floor(a_star*gain),  b_star*gain - np.floor(b_star*gain)

#Defines variable parameter for Chebyshev input
def beta(i):
    global mu
    return np.exp(mu*i*(1-i))


def generateCleanSequence(iterations, x_0, y_0):
    x, y = x_0, y_0

    for j in range(1000):
        x, y = f(x,y)

    iterations -=1000
    xvals, yvals = []
    for j in range(iterations):
        x, y = f(x,y)
        xvals.append(x)
        yvals.append(y)

    return xvals, yvals

def getSubsequence(xsequence, ysequence, k):
    global n

    xseq, yseq =[],[]
    for l in range(k,k+n):
        xseq.append(xsequence[l])
        yseq.append(ysequence[l])

    return xseq, yseq
    # step = K*K

    # newSequence = []
    # for l in range(0,step*n, step):
    #     newSequence.append(sequence[l])

    # return newSequence

def rho(x, y, z):
    return np.sqrt(x*x +y*y+z*z)

def phi():
    pass

def theta():
    pass



def x(rho, phi, theta):
    return rho*np.sin(phi)*np.cos(theta)

def y(rho, phi, theta):
    return rho*np.sin(phi)*np.sin(theta)

def z(rho, phi):
    return rho*np.cos(phi)

def brownianMotion(x,y,z,xStream,yStream):
    global n, r
    for m in range(n):
        pass

    return x_n, y_n, z_n

def main():
    #Read image data
    fileNames = ["","Explosion", "Fence","Ishigami","Pikachu","PowerLines","Shirogane","Tower"]
    fileName = fileNames[4]
    image = open("TestImages/Grey{}.ppm".format(fileName),"r")

    lines = image.readlines()
    dataStream=""
    for i in range(4,len(lines)):
        dataStream+= lines[i]
    image.close()

    # Encode the string to bytes and update the hash
    sha256Hash = hashlib.sha256()
    dataStream = dataStream.encode('utf-8')
    sha256Hash.update(dataStream)
    hexDigest = sha256Hash.hexdigest()

    #Generate the three keys
    key1 = hexDigest[0:32]
    key2 = hexDigest[32:]
    key3 = hex(int(key1, 16) ^ int(key2, 16))
    strKey3 = str(key3)[2:]
    print(strKey3[0:8])

    #Split the key and obtain the 4 plaintext variables
    epsilonValues=[]

    count = 0
    runningH = 0
    for h in strKey3:
        count+=1

        if count==1:
            runningH = int(h,16)
        else:
            runningH = runningH ^ int(h,16)

        if count==8:
            count=0
            runningH /= 15
            epsilonValues.append(runningH)

    print(epsilonValues)

    #Calculate eta and 
    eta = (x_0/y_0) * (mu/k)

    x_0P = (epsilonValues[0]+eta)/(x_0+eta)
    y_0P = (epsilonValues[1]+eta)/(y_0+eta)
    muP  = mu + epsilonValues[2]/eta
    kP   = k + epsilonValues[3]/eta

    print(x_0P, y_0P, muP, kP)

    #Generate Q1 image
    M, N = lines[2].replace("\n","").split(" ")
    low  = min(int(M), int(N))
    hi   = max(int(M), int(N))

    Q_1 = []
    for i in range(4,len(lines)):
        line = lines[i].replace("\n","")
        if line.isnumeric()==False:
            print(i, line)
        Q_1.append(  int( line) )

    print("Len before: ",len(Q_1))
    if low!=hi:
        extension = (hi*hi)-len(Q_1)
        for i in range(extension):
            Q_1.append(0)

    print(low, hi)
    print("Len after: ",len(Q_1))


    #Reshape array into 2D array for coordinates
    K = hi
    gridQ1 = np.array(Q_1).reshape(K,K)
    print(gridQ1)

    coordOnes = []
    for a in range(K):
        for b in range(K):
            coordOnes.append( (a+1,b+1,0))

    #Generate chaotic stream using chaotic map
    global x_0, y_0, mu, k, gain, n

    x_0, y_0, mu, k, gain = x_0P, y_0P, muP, kP, math.pow(10, kP)

    xStream, yStream = generateCleanSequence(K*K*n+1000)

    unnormalizedSeq = []

    for c in range(K*K):
        #Get initial coordinates of this point
        x_A, y_A, z_A = coordOnes[c]

        #Get stream points for this point
        start= c*n
        end  = start+n
        x_AStream = xStream[start:end]
        y_AStream = yStream[start:end]

        x_A, y_A, z_A = brownianMotion(x_A,y_A,z_A,x_AStream,y_AStream)



if __name__=="__main__":
    main()
