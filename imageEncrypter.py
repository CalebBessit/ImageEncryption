#Image Encryption using a the 2D-LCCM
#Caleb Bessit
#05 October 2023

import math
import hashlib
import numpy as np

#INITIAL KEYS
x_0, y_0, mu, k = 1,1,10,10
gain            = math.pow(10,k)
n               = 20
r               = 100
fileName        = "NULL"
useDefault      = True

def f(x, y):
    global mu, gain
    
    a_star = np.cos( beta(y)*np.arccos(x) )
    b_star = np.cos( beta(x)*np.arccos(y) )
    return a_star*gain - np.floor(a_star*gain),  b_star*gain - np.floor(b_star*gain)

#Defines variable parameter for Chebyshev input
def beta(i):
    global mu
    return np.exp(mu*i*(1-i))

#Generates iteration number of terms of the 2D-LCCM map
def generateCleanSequence(iterations, x_0, y_0):
    x, y = x_0, y_0

    for j in range(1000):
        x, y = f(x,y)

    iterations -=1000
    xvals, yvals = [], []
    for j in range(iterations):
        x, y = f(x,y)
        xvals.append(x)
        yvals.append(y)

    return xvals, yvals

#Retrieves n terms of the sequences starting from the kth term
def getSubsequence(xsequence, ysequence, k):
    global n

    xseq, yseq =[],[]
    for l in range(k,k+n):
        xseq.append(xsequence[l])
        yseq.append(ysequence[l])

    return xseq, yseq

#Spherical coordinate angle for Brownian motion
def rho(x_n,y_n):
    return np.mod(  np.floor(  (x_n+y_n)*math.pow(10,8)) ,r+1   )   

#Written in the paper as theta_1
def phi(x_n):
    return np.deg2rad(  np.mod( np.floor(x_n * math.pow(10,8)) ,181  )  )

#Written in the paper as theta_2
def theta(y_n):
    return np.deg2rad(  np.mod(  np.floor(y_n* math.pow(10,8)), 361  )  )


#Conversion functions from spherical to rectangluar coordinates
def x(rho, phi, theta):
    return rho*np.sin(phi)*np.cos(theta)

def y(rho, phi, theta):
    return rho*np.sin(phi)*np.sin(theta)

def z(rho, phi):
    return rho*np.cos(phi)

#Implementation of Brownian motion model using chaotic streams.
#Returns the final position after motion
def brownianMotion(x_n,y_n,z_n,xStream,yStream):
    global n, r
    for m in range(n):
        r_update        = rho(xStream[m],yStream[m])
        theta_1_update  = phi(xStream[m])
        theta_2_update  = theta(yStream[m])

        x_n = x_n + x(r_update, theta_1_update, theta_2_update)
        y_n = y_n + y(r_update, theta_1_update, theta_2_update)
        x_n = z_n + z(r_update, theta_1_update)

    return x_n, y_n, z_n

#Generates two chaotic matrices using chaotic sequences
def generateChaoticMatrices(x2n, y2n,K):
    a1, a2   = [], []
    gain = math.pow(10,8)
    for a in range(K*K):
         #Calculate the values for the chaotic matrices and convert them to 8-bit binary values
         t1 = np.mod( int( (  x2n[a]    + y2n[a]    +1 ) * gain ), 16)
         a1.append( bin(t1 & 0xFF)[2:].zfill(8) )

         t1 = np.mod( int( (  x2n[a]**2 + y2n[a]**2 +1 ) * gain ), 16) 
         a2.append( bin(t1 & 0xFF)[2:].zfill(8)  )

    a1 = np.array(a1).reshape(K,K)
    a2 = np.array(a2).reshape(K,K)

    return a1, a2

def main():
    global x_0, y_0, mu, k, gain, n, fileName, useDefault
    #Read image data

    print("Loading image data...")

    
    #Load from file or use new
    if fileName=="NULL":
        fileNames = ["","Explosion", "Fence","Ishigami","Pikachu","PowerLines","Shirogane","Tower","Heh"]
        fileName = fileNames[4]
        image = open("TestImages/Grey{}.ppm".format(fileName),"r")
    else:
        image = open(fileName, "r")
        useDefault = False


    lines = image.readlines()
    dataStream=""
    for i in range(4,len(lines)):
        dataStream+= lines[i]
    image.close()


    '''Step 3.1'''
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
    print("Generating image hash and system parameters...")
    print("Hash value: {}".format(hexDigest))
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

    # print(epsilonValues)

    #Calculate eta and 
    eta = (x_0/y_0) * (mu/k)

    x_0P = (epsilonValues[0]+eta)/(x_0+eta)
    y_0P = (epsilonValues[1]+eta)/(y_0+eta)
    muP  = mu + epsilonValues[2]/eta
    kP   = k + epsilonValues[3]/eta

    # print(x_0P, y_0P, muP, kP)

    ''' Part 3.2: Step 1'''
    #Generate Q1 image
    M, N = lines[2].replace("\n","").split(" ")
    low  = min(int(M), int(N))
    hi   = max(int(M), int(N))

    print("Generating full image Q1...")

    Q_1 = []
    for i in range(4,len(lines)):
        line = lines[i].replace("\n","")
        # if line.isnumeric()==False:
        #     print(i, line)
        Q_1.append(  int( line) )

    # print("Len before: ",len(Q_1))
    if low!=hi:
        extension = (hi*hi)-len(Q_1)
        for i in range(extension):
            Q_1.append(190)

    # print(low, hi)
    # print("Len after: ",len(Q_1))


    ''' Part 3.2: Step 2'''
    #Reshape array into 2D array for coordinates
    K = hi
    gridQ1 = np.array(Q_1).reshape(K,K)
    # print(gridQ1)

    coordOnes = []
    for a in range(K):
        for b in range(K):
            coordOnes.append( (a+1,b+1,0))

    #Generate chaotic stream using chaotic map
    

    x_0, y_0, mu, k, gain = x_0P, y_0P, muP, kP, math.pow(10, kP)
    print("Generating chaotic sequences...")
    xStream, yStream = generateCleanSequence(K*K*n+1000, x_0, y_0)
    print("Implementing Brownian motion...")
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

        unnormalizedSeq.append(  (x_A, y_A)  )

    '''Part 3.2: Step 4'''

    print("Normalizing data...")
    minX = min(item[0] for item in unnormalizedSeq)
    maxX = max(item[0] for item in unnormalizedSeq)

    minY = min(item[1] for item in unnormalizedSeq)
    maxY = max(item[1] for item in unnormalizedSeq)
    #Begin normalizing values
    xNorm, yNorm = [],[]

    for m in unnormalizedSeq:
        xNorm.append(   (  (m[0]-minX) * (K)    )  /  (maxX-minX)     )
        yNorm.append(   (  (m[1]-minY) * (K)    )  /  (maxY-minY)     )

    print("Generating ranking array...")

    #Suffixed with "X" because it ranks based on x-coordinates
    tempX           = np.array(xNorm).argsort()
    L_primeX        = np.empty_like(tempX)
    L_primeX[tempX] = np.arange(K*K)

    '''Part 3.2: Step 5'''
    #Generate scrambled image. Ranking array acts as a bijective map
    print("Generating scrambled image Q2...")


    tempArr = np.array(Q_1)
    sortedIndices = np.argsort(L_primeX)
    Q_2 = tempArr[sortedIndices]

    Q_2 = Q_2.tolist()

    '''Part 3.3: Rubik's cube transformation'''
    #Get the X_{2n} and Y_{2n} subsequences

    print("Extracting subsequences for encryption...")
    x_2n = xStream[0:K*K]
    y_2n = yStream[0:K*K]

    print("Generating chaotic matrices...")
    A1, A2   = generateChaoticMatrices(x_2n, y_2n, K)
 

    #Reshape scrambled image
    Q_3Bin = np.array(Q_2).reshape(K,K)

    Q_3Hi, Q_3Lo = [],[]

#Iterate over Q2 and convert to binary, split into upper and lower bits,
#store upper and lower halves respectively
    print("Splitting binary values...")
    for g in Q_2:
        binVal = bin(g & 0xFF)[2:].zfill(8) #Convert to binary
        Q_3Hi.append(binVal[0:4])
        Q_3Lo.append(binVal[4:])


    Q_3Hi, Q_3Lo = np.array(Q_3Hi).reshape(K,K),np.array(Q_3Lo).reshape(K,K)


    #Define f(i,j) and g(i,j)

    def F(i,j):
        return int("0b"+Q_3Lo[i][j],2) ^ int("0b"+A1[i][j],2) ^ int("0b"+A2[K-1-i][K-1-j],2)

    def G(i,j):
        return int("0b"+Q_3Hi[i][j],2) ^ int("0b"+A1[K-1-i][K-1-j],2) ^ int("0b"+A2[i][j],2)


    #Iterate and find Q3H' and Q3L' by diffusing using Henon map
    precision = math.pow(10,8)
    k_0, k_1  = 1, 1

    print("Starting diffusion...")
    Q_3HiPri, Q_3LoPri = np.zeros(K*K).reshape(K,K),np.zeros(K*K).reshape(K,K)
    for o in range(K):
        for p in range(K):
            
            #Lower
            if (o==0 and p==0):
                Q_3LoPri[o][p] = int(F(o,p)) ^  int (np.mod( np.floor( (1-1.4 * (k_0/15)**2 + (k_1/15)) *precision ) ,16))
                Q_3HiPri[o][p] = int(G(o,p)) ^ int(np.mod( np.floor(0.3 * (k_0/15) *precision)  ,16))
            elif (o!=0 and p==0):
                Q_3LoPri[o][p] = int(F(o,p)) ^ int(np.mod( np.floor( (1-1.4 * (Q_3LoPri[o-1][p-1]/15)**2 + (Q_3HiPri[o][p-1]/15)) *precision ) ,16))
                Q_3HiPri[o][p] = int(G(o,p)) ^ int(np.mod( np.floor(0.3 * (Q_3LoPri[o-1][K-1]/15) *precision) ,16))
            elif (p!=0):
                Q_3LoPri[o][p] = int(F(o,p)) ^ int(np.mod( np.floor( (1-1.4 * (Q_3LoPri[o][p-1]/15)**2 + (Q_3HiPri[o][p-1]/15)) *precision ) ,16))
                Q_3HiPri[o][p] = int(G(o,p)) ^ int(np.mod( np.floor(0.3 * (Q_3LoPri[o][p-1]/15) *precision) ,16))
            


    #Recombine encrypted matrices
    Q_3HiPri = Q_3HiPri.reshape(1,K*K)[0].tolist()
    Q_3LoPri = Q_3LoPri.reshape(1,K*K)[0].tolist()

    Q_4 = []

    for q in range(len(Q_3HiPri)):
        value = "0b" + bin(int(Q_3HiPri[q]))[2:].zfill(4) + bin(int(Q_3LoPri[q]))[2:].zfill(4)
        Q_4.append( str(int(value, 2)) +"\n" )

    print("Diffusion complete.")
    print("Saving encrypted image to file...")

    fileHeader = "P2\n# Encrypted Image\n{} {}\n255\n".format(K,K)

    fileContent = "".join(Q_4)
    fileContent = fileHeader + fileContent

    if useDefault:
        scrambledImage = open("TestImages/GreyEncrypted{}.ppm".format(fileName),"w")
    else:
        scrambledImage = open("GreyEncrypted{}.ppm".format(fileName),"w")

    scrambledImage.write(fileContent)
    scrambledImage.close()

    print("Done.")
   
    

if __name__=="__main__":
    main()
