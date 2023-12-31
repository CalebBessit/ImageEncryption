#Image Encryption using a the 2D-LCCM: Color image version
#Caleb Bessit
#15 December 2023

import math
import hashlib
import numpy as np
import multiprocessing


#INITIAL KEYS
x_0, y_0, mu, k     = 1,1,8,8
x_0S, y_0S, muS, kS = x_0, y_0, mu, k  
gain                = math.pow(10,k)
n                   = 20
r                   = 100
fileName            = "NULL"
useDefault          = True

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

    xS, yS          = np.array(xStream), np.array(yStream)
    r_update        = rho(xS,yS)
    theta_1_update  = phi(xS)
    theta_2_update  = theta(yS)

    updateX = x(r_update, theta_1_update, theta_2_update)
    updateY = y(r_update, theta_1_update, theta_2_update)
   
    x_n = n*x_n + np.sum(updateX)
    y_n = n*y_n + np.sum(updateY)
    return x_n, y_n

def eightBitStringify(x):
    return bin(x & 0xFF)[2:].zfill(8)

#Generates two chaotic matrices using chaotic sequences
def generateChaoticMatrices(x2n, y2n,K):
    a1, a2   = [], []
    gain = math.pow(10,8)
    for a in range(K*K):
         #Calculate the values for the chaotic matrices and convert them to 8-bit binary values
         t1 = np.mod( int( (  x2n[a]    + y2n[a]    +1 ) * gain ), 16)
         a1.append( bin(t1 & 0xFF)[2:].zfill(8) )

         t2 = np.mod( int( (  x2n[a]**2 + y2n[a]**2 +1 ) * gain ), 16) 
         a2.append( bin(t2 & 0xFF)[2:].zfill(8)  )

    # x, y = np.array(x2n), np.array(y2n)

    # x, y = (x+y+1)*gain, (x**2+y**2+1)*gain

    # x, y = x.astype(int), y.astype(int)

    # x, y = list(np.mod(x, 16)), list(np.mod(y, 16))

    # x = [eightBitStringify(a) for a in x]
    # y = [eightBitStringify(b) for b in y]


    a1 = np.array(a1).reshape(K,K)
    a2 = np.array(a2).reshape(K,K)

    return a1, a2

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


def processData(coordOnes, xStream, yStream, n, c, unnormalizedSeq):
    x_A, y_A, z_A = coordOnes[c]
    start = c * n
    end = start + n
    x_AStream = xStream[start:end]
    y_AStream = yStream[start:end]
    x_A, y_A, z_A = brownianMotion(x_A, y_A, z_A, x_AStream, y_AStream)
    unnormalizedSeq[c] = (x_A, y_A)


def main():
    global x_0, y_0, mu, k, gain, n, fileName, useDefault
    #Read image data

    print("Loading image data...")

    
    #Load from file or use new
    if fileName=="NULL":
        fileNames = ["Test","Explosion", "Fence","Ishigami","Pikachu","PowerLines","Shirogane","Tower","Heh"]
        fileName = fileNames[4]
        image = open("TestImages/{}.ppm".format(fileName),"r")
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

    #Calculate eta and other initial keys
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
        Q_1.append(  int( line) )

    R1, G1, B1 = Q_1[::3],Q_1[1::3],Q_1[2::3]

    if low!=hi:
        extension = (hi*hi)-len(R1)
        for i in range(extension):
            R1.append(0)
            G1.append(0)
            B1.append(0)


    ''' Part 3.2: Step 2'''
    #Reshape array into 2D array for coordinates
    K = hi
    coordOnes = []
    for a in range(K):
        for b in range(K):
            coordOnes.append( (a+1,b+1,0))

    #Generate chaotic stream using chaotic map
    

    x_0, y_0, mu, k, gain = x_0P, y_0P, muP, kP, math.pow(10, kP)
    print("Generating chaotic sequences...")
    xStream, yStream = generateCleanSequence(K*K*n+1000, x_0, y_0)
    print("Implementing parallelized Brownian motion...")
    
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


    tR, tG, tB = np.array(R1), np.array(G1), np.array(B1)
    sortedIndices = np.argsort(L_primeX)
    R2, G2, B2 = tR[sortedIndices],tG[sortedIndices],tB[sortedIndices]

    R2, G2, B2 = R2.tolist(), G2.tolist(), B2.tolist()

    '''Part 3.3: Rubik's cube transformation'''
    #Get the X_{2n} and Y_{2n} subsequences

    print("Extracting subsequences for encryption...")
    x_2n = xStream[0:K*K]
    y_2n = yStream[0:K*K]

    print("Generating all chaotic matrices...")
    A1, A2   = generateChaoticMatrices(x_2n, y_2n, K)
 
    S1 = generateSecondaryChaoticMatrices(x_2n, y_2n,2,2,K)
    S2 = generateSecondaryChaoticMatrices(x_2n, y_2n,1,2,K)
    S3 = generateSecondaryChaoticMatrices(x_2n, y_2n,2,1,K)
    S4 = generateSecondaryChaoticMatrices(x_2n, y_2n,1,1,K)
    S5 = generateSecondaryChaoticMatrices(x_2n, y_2n,2,3,K)

    xSub, ySub = x_2n[0:1000], y_2n[0:1000]

    S6 = binaryMask(xSub)   #0=Row rotation, 1=column rotation
    S7 = binaryMask(ySub)   #0=left/up, 1=right/down

    S8 = generateTernaryChaoticMatrices(xSub,K)
    S9 = generateTernaryChaoticMatrices(ySub,4)


    sR, sG, sB = np.array(R2).reshape((K,K)), np.array(G2).reshape((K,K)), np.array(B2).reshape((K,K))
    print("Splicing into and scrambling virtual Rubik's cube...")


    sR,sr1,sr2,sr3,sr4,sr5 = scrambleRubiksCube(sR,S1,S2,S3,S4,S5,S6,S7,S8,S9)
    sG,sg1,sg2,sg3,sg4,sg5 = scrambleRubiksCube(sG,S1,S2,S3,S4,S5,S6,S7,S8,S9)
    sB,sb1,sb2,sb3,sb4,sb5 = scrambleRubiksCube(sB,S1,S2,S3,S4,S5,S6,S7,S8,S9)

    R2, G2, B2 = list(sR.reshape(1,K*K)[0]), list(sG.reshape(1,K*K)[0]), list(sB.reshape(1,K*K)[0])
   
    print("Scrambling complete.")
    #Arrays for binary values
    Q_3Hi, Q_3Lo = [],[]
    
    R3Hi, R3Lo, G3Hi, G3Lo, B3Hi, B3Lo = [],[], [],[], [],[]
    
    #Iterate over Q2 and convert to binary, split into upper and lower bits,
    #store upper and lower halves respectively
    print("Splitting binary values...")
    for i in range(len(R2)):
        binVal = bin(R2[i] & 0xFF)[2:].zfill(8) #Convert to binary
        R3Hi.append(binVal[0:4])
        R3Lo.append(binVal[4:])

        binVal = bin(G2[i] & 0xFF)[2:].zfill(8) #Convert to binary
        G3Hi.append(binVal[0:4])
        G3Lo.append(binVal[4:])


        binVal = bin(B2[i] & 0xFF)[2:].zfill(8) #Convert to binary
        B3Hi.append(binVal[0:4])
        B3Lo.append(binVal[4:])



    R3Hi, R3Lo = np.array(R3Hi).reshape(K,K),np.array(R3Lo).reshape(K,K)
    G3Hi, G3Lo = np.array(G3Hi).reshape(K,K),np.array(G3Lo).reshape(K,K)
    B3Hi, B3Lo = np.array(B3Hi).reshape(K,K),np.array(B3Lo).reshape(K,K)

    #Define f(i,j) and g(i,j) for each color

    def Fr(i,j):
        return int("0b"+R3Lo[i][j],2) ^ int("0b"+A1[i][j],2) ^ int("0b"+A2[K-1-i][K-1-j],2)

    def Gr(i,j):
        return int("0b"+R3Hi[i][j],2) ^ int("0b"+A1[K-1-i][K-1-j],2) ^ int("0b"+A2[i][j],2)


    def Fg(i,j):
        return int("0b"+G3Lo[i][j],2) ^ int("0b"+A1[i][j],2) ^ int("0b"+A2[K-1-i][K-1-j],2)

    def Gg(i,j):
        return int("0b"+G3Hi[i][j],2) ^ int("0b"+A1[K-1-i][K-1-j],2) ^ int("0b"+A2[i][j],2)


    def Fb(i,j):
        return int("0b"+B3Lo[i][j],2) ^ int("0b"+A1[i][j],2) ^ int("0b"+A2[K-1-i][K-1-j],2)

    def Gb(i,j):
        return int("0b"+B3Hi[i][j],2) ^ int("0b"+A1[K-1-i][K-1-j],2) ^ int("0b"+A2[i][j],2)


    #Iterate and find Q3H' and Q3L' by diffusing using Henon map
    precision = math.pow(10,8)
    k_0, k_1  = 1, 1

    print("Starting diffusion...")
    R3HiPri, R3LoPri = np.zeros(K*K).reshape(K,K),np.zeros(K*K).reshape(K,K)
    G3HiPri, G3LoPri = np.zeros(K*K).reshape(K,K),np.zeros(K*K).reshape(K,K)
    B3HiPri, B3LoPri = np.zeros(K*K).reshape(K,K),np.zeros(K*K).reshape(K,K)
    for o in range(K):
        for p in range(K):
            
            #Lower
            if (o==0 and p==0):
                R3LoPri[o][p] = int(Fr(o,p)) ^  int (np.mod( int( (1-1.4 * (k_0/15)**2 + (k_1/15)) *precision ) ,16))
                R3HiPri[o][p] = int(Gr(o,p)) ^ int(np.mod( int(0.3 * (k_0/15) *precision)  ,16))

                G3LoPri[o][p] = int(Fg(o,p)) ^  int (np.mod( int( (1-1.4 * (k_0/15)**2 + (k_1/15)) *precision ) ,16))
                G3HiPri[o][p] = int(Gg(o,p)) ^ int(np.mod( int(0.3 * (k_0/15) *precision)  ,16))

                B3LoPri[o][p] = int(Fb(o,p)) ^  int (np.mod( int( (1-1.4 * (k_0/15)**2 + (k_1/15)) *precision ) ,16))
                B3HiPri[o][p] = int(Gb(o,p)) ^ int(np.mod( int(0.3 * (k_0/15) *precision)  ,16))
            elif (o!=0 and p==0):
                R3LoPri[o][p] = int(Fr(o,p)) ^ int(np.mod( int( (1-1.4 * (R3LoPri[o-1][K-1]/15)**2 + (R3HiPri[o-1][K-1]/15)) *precision ) ,16))
                R3HiPri[o][p] = int(Gr(o,p)) ^ int(np.mod( int(0.3 * (R3LoPri[o-1][K-1]/15) *precision) ,16))

                G3LoPri[o][p] = int(Fg(o,p)) ^ int(np.mod( int( (1-1.4 * (G3LoPri[o-1][K-1]/15)**2 + (G3HiPri[o-1][K-1]/15)) *precision ) ,16))
                G3HiPri[o][p] = int(Gg(o,p)) ^ int(np.mod( int(0.3 * (G3LoPri[o-1][K-1]/15) *precision) ,16))

                B3LoPri[o][p] = int(Fb(o,p)) ^ int(np.mod( int( (1-1.4 * (B3LoPri[o-1][K-1]/15)**2 + (B3HiPri[o-1][K-1]/15)) *precision ) ,16))
                B3HiPri[o][p] = int(Gb(o,p)) ^ int(np.mod( int(0.3 * (B3LoPri[o-1][K-1]/15) *precision) ,16))
            elif (p!=0):
                R3LoPri[o][p] = int(Fr(o,p)) ^ int(np.mod( int( (1-1.4 * (R3LoPri[o][p-1]/15)**2 + (R3HiPri[o][p-1]/15)) *precision ) ,16))
                R3HiPri[o][p] = int(Gr(o,p)) ^ int(np.mod( int(0.3 * (R3LoPri[o][p-1]/15) *precision) ,16))

                G3LoPri[o][p] = int(Fg(o,p)) ^ int(np.mod( int( (1-1.4 * (G3LoPri[o][p-1]/15)**2 + (G3HiPri[o][p-1]/15)) *precision ) ,16))
                G3HiPri[o][p] = int(Gg(o,p)) ^ int(np.mod( int(0.3 * (G3LoPri[o][p-1]/15) *precision) ,16))

                B3LoPri[o][p] = int(Fb(o,p)) ^ int(np.mod( int( (1-1.4 * (B3LoPri[o][p-1]/15)**2 + (B3HiPri[o][p-1]/15)) *precision ) ,16))
                B3HiPri[o][p] = int(Gb(o,p)) ^ int(np.mod( int(0.3 * (B3LoPri[o][p-1]/15) *precision) ,16))
            


    #Recombine encrypted matrices
    R3HiPri = R3HiPri.reshape(1,K*K)[0].tolist()
    R3LoPri = R3LoPri.reshape(1,K*K)[0].tolist()

    G3HiPri = G3HiPri.reshape(1,K*K)[0].tolist()
    G3LoPri = G3LoPri.reshape(1,K*K)[0].tolist()

    B3HiPri = B3HiPri.reshape(1,K*K)[0].tolist()
    B3LoPri = B3LoPri.reshape(1,K*K)[0].tolist()

    Q_4 = []

    for q in range(len(R3HiPri)):
        value1 = "0b" + bin(int(R3HiPri[q]))[2:].zfill(4) + bin(int(R3LoPri[q]))[2:].zfill(4)
        value2 = "0b" + bin(int(G3HiPri[q]))[2:].zfill(4) + bin(int(G3LoPri[q]))[2:].zfill(4)
        value3 = "0b" + bin(int(B3HiPri[q]))[2:].zfill(4) + bin(int(B3LoPri[q]))[2:].zfill(4)

        Q_4.append( str(int(value1, 2)) +"\n" )
        Q_4.append( str(int(value2, 2)) +"\n" )
        Q_4.append( str(int(value3, 2)) +"\n" )
            

    print("Diffusion complete.")
    print("Saving encrypted image to file...")

    #Save all relevant data to file
    fileHeader = "P3\n# Encrypted Color Image\n{} {}\n255\n".format(K,K)

    fileContent = "".join(Q_4)
    fileContent = fileHeader + fileContent

    if useDefault:
        scrambledImage = open("TestImages/ColorEncrypted{}.ppm".format(fileName),"w")
    else:
        scrambledImage = open("ColorEncrypted{}.ppm".format(fileName),"w")

    scrambledImage.write(fileContent)
    scrambledImage.close()

    print("Saving decryption data to file...")

    np.save("DecryptionData/Color{}.npy".format(fileName), [[sr1,sr2,sr3,sr4,sr5],[sg1,sg2,sg3,sg4,sg5],[sb1,sb2,sb3,sb4,sb5]])
    file = open("DecryptionData/Color{}.txt".format(fileName),"w")
    print("Hash Code: {}".format(hexDigest),file=file)
    print("Initial keys <x_0, y_0, mu, k>: {}, {}, {}, {}".format(x_0S, y_0S, muS, kS),file=file)

    
    file.close()
    print("Done.")
   
    

if __name__=="__main__":
    main()
