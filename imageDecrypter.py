#Image decrypter for encrypted images
#Caleb Bessit
#08 October 2023

import os
import math
import time
import numpy as np

#INITIAL KEYS
x_0, y_0, mu, k = 1,1,1,1
gain            = math.pow(10,k)
n               = 20
r               = 100
hexDigest       = "NULL"
fileName        = "NULL"

def f(x, y):
    global mu, gain
    
    a_star = math.cos( beta(y)*math.acos(x) )
    b_star = math.cos( beta(x)*math.acos(y) )
    return a_star*gain - math.floor(a_star*gain),  b_star*gain - math.floor(b_star*gain)

#Defines variable parameter for Chebyshev input
def beta(i):
    global mu
    return math.exp(mu*i*(1-i))


#Generates iteration number of terms of the 2D-LCCM map
def generateCleanSequence(iterations, x_0, y_0):
    x, y = x_0, y_0

    for j in range(1000):
        x, y = f(x,y)

    iterations -=1000
    xvals, yvals = np.empty(iterations), np.empty(iterations)
    for j in range(iterations):
        x, y = f(x,y)
        xvals[j], yvals[j] = x, y
    return xvals, yvals

def rho(x_n,y_n):
    return np.mod(  np.floor(  (x_n+y_n)*math.pow(10,8)) ,r+1   )   

#Written in the paper as theta_1
def phi(x_n):
    return np.deg2rad(  np.mod( np.floor(x_n * math.pow(10,8)) ,181  )  )

#Written in the paper as theta_2
def theta(y_n):
    return np.deg2rad(  np.mod(  np.floor(y_n* math.pow(10,8)), 361  )  )



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

    xS, yS          = np.array(xStream), np.array(yStream)
    r_update        = rho(xS,yS)
    theta_1_update  = phi(xS)
    theta_2_update  = theta(yS)

    updateX = x(r_update, theta_1_update, theta_2_update)
    updateY = y(r_update, theta_1_update, theta_2_update)
    x_n = n*x_n + math.fsum(updateX)
    y_n = n*y_n + math.fsum(updateY)
    return x_n, y_n

def eightBitStringify(x):
    return bin(x & 0xFF)[2:].zfill(8)

def generateChaoticMatrices(x2n, y2n,K):
    gain = math.pow(10,8)
    
    x, y = np.array(x2n), np.array(y2n)

    x, y = (x+y+1)*gain, (x**2+y**2+1)*gain

    x, y = x.astype(int), y.astype(int)

    x, y = list(np.mod(x, 16)), list(np.mod(y, 16))

    x = [eightBitStringify(a) for a in x]
    y = [eightBitStringify(b) for b in y]


    x = np.array(x).reshape(K,K)
    y = np.array(y).reshape(K,K)

    return x,y

#Generates a chaotic matrix based on the rule x2n^(p1)-y2n^(p2)+1
def generateSecondaryChaoticMatrices(x2n,y2n,p1,p2,K):
    gain = math.pow(10,8)
    x = np.array(x2n)
    y = np.array(y2n)

    s = (x**p1 - y**p2 +1)*gain
    s = s.astype(int)
    s = np.mod(s,256)

    return s.reshape((K,K))

def generateTernaryChaoticMatrices(sequence,modulus):
    gain = math.pow(10,8)
    x = sequence[0:1000]
    x = np.array(x)*gain
    x = x.astype(int)
    x = np.mod(x,modulus)

    return list(x)


def binaryMask(subsequence):
    x = np.array(subsequence)
    return np.where(x>0.5,1,0)


def rotateRow(matrices, index):
    temps = []
    lastRow = matrices[-1][index].copy()
    for m in range(len(matrices)-1,0,-1):
        #Make a copy of row of interest in previous matrix
        #Make copy of current matrix
        #Exchange current row with row of interest, add to list
        tempRow     = matrices[m-1][index].copy()
        temp        = matrices[m].copy()
        temp[index] = tempRow
        temps.append(temp)

    #Do for last index to "wrap around"
    temp = matrices[0].copy()
    temp[index] = lastRow
    temps.append(temp)

    #Correct list order
    return list(reversed(temps))

def rotateColumn(matrices, index):
    temp = []
    for m in matrices:
        temp.append(m.copy().T)
    
    exchangedTemp = rotateRow(temp, index)

    result = []
    for e in exchangedTemp:
        result.append(e.copy().T)
    
    return result

def scrambleRubiksCube(f1, f2, f3, f4, f5, f6, rOC,direction,index,extent):

    m1, m2, m3, m4, m5, m6 = f1.copy(), f2.copy(), f3.copy(), f4.copy(), f5.copy(), f6.copy()
    
    for i in range(len(extent)):
        ext = int(extent[i])+1
        if ext==4:
            continue
        elif ext==2:
            #Double rotation: use two opposing faces

            #Rotate a row
            if int(rOC[i]==0):
                m1, m6 = rotateRow([m1,m6],int(index[i]))
                m3, m5 = rotateRow([m3,m5],int(index[i]))
            else:
                #Rotate a column
                m1, m6 = rotateColumn([m1,m6],int(index[i]))
                m2, m4 = rotateColumn([m2,m4],int(index[i]))
        else:
            #Single rotation. Here, direction matters
            #If it is a rotation by 3, flip the direction and rotate by one
            direc = int(direction[i])
            if int(extent[i]==3):
                direc = 1 if (direc==0) else 0

            if int(rOC[i]==0):
                if direc==0:
                    m1, m5, m6, m3 = rotateRow([m1,m5,m6,m3],int(index[i]))
                else:
                    m3, m6, m5, m1 = rotateRow([m3,m6,m5,m1],int(index[i]))
                
            else:
                if direc==0:
                    m1, m2, m6, m4 = rotateColumn([m1, m2, m6, m4],int(index[i]))
                else:
                    m4, m6, m2, m1 = rotateColumn([m4, m6, m2, m1],int(index[i]))

    
    return [m1, m2, m3, m4, m5, m6]



def main():
    global x_0, y_0, mu, k, gain, n, hexDigest, fileName, useDefault
    #Read image data
    
    fileName    = input("Enter the relative path to the file:")

    try:
        image       = open(fileName,"r")
    except Exception:
        print("Image file does not exist/path incorrect.")
        exit()

    fileNoPath  = os.path.basename(fileName)
    fileNoPath  = fileNoPath[:-4]

    pre = time.time_ns()
    

    try:
        decData         = open("DecryptionData/{}.txt".format(fileNoPath),"r")
    except Exception:
        print("Necessary decryption data missing. Please re-try or re-encrypt the image, without deleting or modifying any of the files created.")
        exit()

    print("Loading image data...")
    lines           = decData.readlines()
    hexDigest       = lines[0]
    hexDigest       = hexDigest[hexDigest.rfind(":")+2:]
    keyData         = lines[1]
    keyData         = keyData[keyData.rfind(":")+2:] 
    x_0, y_0, mu, k = list(map(int, keyData.split(",")))

    decData.close()
    
    lines = image.readlines()
    image.close()

    #Generate the three keys
    key1 = hexDigest[0:32]
    key2 = hexDigest[32:]
    key3 = hex(int(key1, 16) ^ int(key2, 16))
    strKey3 = str(key3)[2:]
    print("Generating image hash and system parameters...")

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
    K = hi
    print("Generating full image Q1...")

    imageLines = lines[4:]
    Q_1 = [int(  line.replace("\n","")  )  for line in imageLines]


    #Generate chaotic stream using chaotic map
    

    x_0, y_0, mu, k, gain = x_0P, y_0P, muP, kP, math.pow(10, kP)
    print("Generating chaotic sequences...")
    xStream, yStream = generateCleanSequence(K*K*n+1000, x_0, y_0)

    print("Extracting subsequences for decryption...")
    x_2n = xStream[0:K*K]
    y_2n = yStream[0:K*K]

    print("Generating chaotic matrices...")
    A1, A2   = generateChaoticMatrices(x_2n, y_2n, K)

    Q_2Hi, Q_2Lo = [],[]

    #Iterate over Q2 and convert to binary, split into upper and lower bits,
    #store upper and lower halves respectively
    print("Splitting binary values...")
    for g in Q_1:
        binVal = bin(g & 0xFF)[2:].zfill(8) #Convert to binary
        Q_2Hi.append(binVal[0:4])
        Q_2Lo.append(binVal[4:])


    Q_2Hi, Q_2Lo = np.array(Q_2Hi).reshape(K,K),np.array(Q_2Lo).reshape(K,K)


    def F(i,j):
        return int("0b"+Q_2Lo[i][j],2) ^ int("0b"+A1[i][j],2) ^ int("0b"+A2[K-1-i][K-1-j],2)

    def G(i,j):
        return int("0b"+Q_2Hi[i][j],2) ^ int("0b"+A1[K-1-i][K-1-j],2) ^ int("0b"+A2[i][j],2)


    #Iterate and find Q2H' and Q2L' by diffusing using Henon map
    precision = math.pow(10,8)
    k_0, k_1  = 1, 1

    Q_2HiPri, Q_2LoPri = np.zeros(K*K).reshape(K,K),np.zeros(K*K).reshape(K,K)
    for o in range(K-1,-1,-1):
        for p in range(K-1,-1,-1):
            #Lower
            if (o==0 and p==0):
                Q_2LoPri[o][p] = int(F(o,p)) ^  int ( math.floor( (1-1.4 * (k_0/15)**2 + (k_1/15)) *precision ) %16)
                Q_2HiPri[o][p] = int(G(o,p)) ^ int( math.floor(0.3 * (k_0/15) *precision)  %16)
            elif (o!=0 and p==0):
                Q_2LoPri[o][p] = int(F(o,p)) ^ int( math.floor( (1-1.4 * (int("0b"+Q_2Lo[o-1][K-1],2)/15)**2 + (int("0b"+Q_2Hi[o-1][K-1],2)/15)) *precision ) %16)
                Q_2HiPri[o][p] = int(G(o,p)) ^ int( math.floor(0.3 * (int("0b"+Q_2Lo[o-1][K-1],2)/15) *precision) %16)
            elif (p!=0):
                Q_2LoPri[o][p] = int(F(o,p)) ^ int( math.floor( (1-1.4 * (int("0b"+Q_2Lo[o][p-1],2)/15)**2 + (int("0b"+Q_2Hi[o][p-1],2)/15)) *precision ) %16)
                Q_2HiPri[o][p] = int(G(o,p)) ^ int( math.floor(0.3 * (int("0b"+Q_2Lo[o][p-1],2)/15) *precision) %16)
            
           
                
            

    #Recombine encrypted matrices
    Q_2HiPri = Q_2HiPri.reshape(1,K*K)[0].tolist()
    Q_2LoPri = Q_2LoPri.reshape(1,K*K)[0].tolist()

    Q_2 = []

    for q in range(len(Q_2HiPri)):
        value = "0b" + bin(int(Q_2HiPri[q]))[2:].zfill(4) + bin(int(Q_2LoPri[q]))[2:].zfill(4)
        Q_2.append(int(value,2))

   

    ''' Part 3.2: Step 2'''
    #Reshape array into 2D array for coordinates
    
    try:
        loadedArray = np.load("DecryptionData/{}.npy".format(fileNoPath),allow_pickle=True)
    except Exception:
        print("Necessary decryption data missing. Please re-try or re-encrypt the image, without deleting or modifying any of the files created.")
        exit()

    print("Retrieving decryption arrays from file...")
    S1, S2, S3, S4, S5 = loadedArray
    print("Splicing into and unscrambling virtual Rubik's cube...")

    xSub, ySub = x_2n[0:1000], y_2n[0:1000]

    S6 = binaryMask(xSub)   #0=Row rotation, 1=column rotation
    S7 = binaryMask(ySub)   #0=left/up, 1=right/down

    S8 = generateTernaryChaoticMatrices(xSub,K)
    S9 = generateTernaryChaoticMatrices(ySub,4)

    #Reverse to undo
    S6, S7, S8, S9 = list(reversed(S6)), list(reversed(S7)), list(reversed(S8)), list(reversed(S9))

    S7 = [l^1 for l in S7]

    S0 = np.array(Q_2).reshape((K,K))

    S0,S1,S2,S3,S4,S5 = scrambleRubiksCube(S0,S1,S2,S3,S4,S5,S6,S7,S8,S9)
    Q_2 = list(S0.reshape(1,K*K)[0])

    print("Done with Rubik's cube transformation.")
    coordOnes = [(a + 1, b + 1, 0) for a in range(K) for b in range(K)]
    
    print("Implementing Brownian motion...")

    unnormalizedSeq = list(np.zeros(K*K))

    streamListX, streamListY = np.array(xStream).reshape(-1,n).tolist(), np.array(yStream).reshape(-1,n).tolist()

    unnormalizedSeq = [
        brownianMotion(x, y, z, streamListX[c], streamListY[c])
        for c, (x, y, z) in enumerate(coordOnes)
    ]

    '''Part 3.2: Step 4'''

    print("Normalizing data...")
    minX = min(item[0] for item in unnormalizedSeq)
    maxX = max(item[0] for item in unnormalizedSeq)

    xNorm = [   (  (m[0]-minX) * (K)    )  /  (maxX-minX) for m in unnormalizedSeq  ]
    
    print("Generating ranking array...")
    tempX           = np.array(xNorm).argsort()
    L_primeX        = np.empty_like(tempX)
    L_primeX[tempX] = np.arange(K*K)

    print("Reversing ranking map...")

    p        = np.asanyarray(L_primeX)
    s        = np.empty_like(p)
    s[p]     = np.arange(p.size)
    L_primeX = s

    print("Unscrambling image Q2 -> Q1...")
    tempArr = np.array(Q_2)
    sortedIndices = np.argsort(L_primeX)
    Q_0 = tempArr[sortedIndices]

    Q_0 = Q_0.tolist()

    print("Saving decrypted image to file...")

    fileHeader = "P2\n# Decrypted Image\n{} {}\n255\n".format(K,K)

    Q_0 = [str(int(x))+"\n" for x in Q_0]
  
    fileContent = "".join(Q_0)
    fileContent = fileHeader + fileContent

    decryptedImagePath = "decryptedImages/{}.ppm".format(fileNoPath)
    os.makedirs(os.path.dirname(decryptedImagePath), exist_ok=True)

    decryptedImage = open(decryptedImagePath,"w")
    decryptedImage.write(fileContent)
    decryptedImage.close()

    print("Done.")
    post = time.time_ns()
    print("Total time to decrypt image: {} seconds".format((post-pre)*math.pow(10,-9)))

if __name__ == "__main__":
    main()
