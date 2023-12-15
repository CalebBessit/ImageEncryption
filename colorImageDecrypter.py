#Image decrypter for encrypted images: Color version
#Caleb Bessit
#15 December 2023

import math
import numpy as np

#INITIAL KEYS
x_0, y_0, mu, k = 1,1,1,1
gain            = math.pow(10,k)
n               = 20
r               = 100
hexDigest       = "NULL"
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
    # for m in range(n):
    #     r_update        = rho(xStream[m],yStream[m])
    #     theta_1_update  = phi(xStream[m])
    #     theta_2_update  = theta(yStream[m])

    #     x_n = x_n + x(r_update, theta_1_update, theta_2_update)
    #     y_n = y_n + y(r_update, theta_1_update, theta_2_update)
    #     x_n = z_n + z(r_update, theta_1_update)
    x_n = n*x_n + np.sum(updateX)
    y_n = n*y_n + np.sum(updateY)
    return x_n, y_n

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

    print("Loading image data...")

    if fileName=="NULL":
        fileNames = ["Test","Explosion", "Fence","Ishigami","Pikachu","PowerLines","Shirogane","Tower","Heh"]
        fileIndex = 4
        fileName = fileNames[fileIndex]
        image = open("TestImages/ColorEncrypted{}.ppm".format(fileName),"r")
    else:
        image = open(fileName, "r")
    
    decData         = open("DecryptionData/Color{}.txt".format(fileName),"r")
    lines           = decData.readlines()
    hexDigest       = lines[0]
    hexDigest       = hexDigest[hexDigest.rfind(":")+2:]
    keyData         = lines[1]
    keyData         = keyData[keyData.rfind(":")+2:] 
    x_0, y_0, mu, k = list(map(int, keyData.split(",")))

    decData.close()
    
    lines = image.readlines()
    dataStream=""
    for i in range(4,len(lines)):
        dataStream+= lines[i]
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
    # low  = min(int(M), int(N))
    # hi   = max(int(M), int(N))
    K = int(M) #Dimensions are the same since the encrypted image is made into a square image
    print("Generating full image Q1...")

    Q_1 = []
    for i in range(4,len(lines)):
        line = lines[i].replace("\n","")
        Q_1.append(  int( line) )


    R1, G1, B1 = Q_1[::3],Q_1[1::3],Q_1[2::3]


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

    R2Hi, R2Lo, G2Hi, G2Lo, B2Hi, B2Lo = [], [], [], [], [], []

    #Iterate over encrypted image and convert to binary, split into upper and lower bits,
    #store upper and lower halves respectively
    print("Splitting binary values...")
    for i in range(len(R1)):
        binVal = bin(R1[i] & 0xFF)[2:].zfill(8) #Convert to binary
        R2Hi.append(binVal[0:4])
        R2Lo.append(binVal[4:])

        binVal = bin(G1[i] & 0xFF)[2:].zfill(8) #Convert to binary
        G2Hi.append(binVal[0:4])
        G2Lo.append(binVal[4:])


        binVal = bin(B1[i] & 0xFF)[2:].zfill(8) #Convert to binary
        B2Hi.append(binVal[0:4])
        B2Lo.append(binVal[4:])


    R2Hi, R2Lo = np.array(R2Hi).reshape(K,K),np.array(R2Lo).reshape(K,K)
    G2Hi, G2Lo = np.array(G2Hi).reshape(K,K),np.array(G2Lo).reshape(K,K)
    B2Hi, B2Lo = np.array(B2Hi).reshape(K,K),np.array(B2Lo).reshape(K,K)


    def Fr(i,j):
        return int("0b"+R2Lo[i][j],2) ^ int("0b"+A1[i][j],2) ^ int("0b"+A2[K-1-i][K-1-j],2)

    def Gr(i,j):
        return int("0b"+R2Hi[i][j],2) ^ int("0b"+A1[K-1-i][K-1-j],2) ^ int("0b"+A2[i][j],2)


    def Fg(i,j):
        return int("0b"+G2Lo[i][j],2) ^ int("0b"+A1[i][j],2) ^ int("0b"+A2[K-1-i][K-1-j],2)

    def Gg(i,j):
        return int("0b"+G2Hi[i][j],2) ^ int("0b"+A1[K-1-i][K-1-j],2) ^ int("0b"+A2[i][j],2)


    def Fb(i,j):
        return int("0b"+B2Lo[i][j],2) ^ int("0b"+A1[i][j],2) ^ int("0b"+A2[K-1-i][K-1-j],2)

    def Gb(i,j):
        return int("0b"+B2Hi[i][j],2) ^ int("0b"+A1[K-1-i][K-1-j],2) ^ int("0b"+A2[i][j],2)


    #Iterate and find Q2H' and Q2L' by diffusing using Henon map
    precision = math.pow(10,8)
    k_0, k_1  = 1, 1

    R2HiPri, R2LoPri = np.zeros(K*K).reshape(K,K),np.zeros(K*K).reshape(K,K)
    G2HiPri, G2LoPri = np.zeros(K*K).reshape(K,K),np.zeros(K*K).reshape(K,K)
    B2HiPri, B2LoPri = np.zeros(K*K).reshape(K,K),np.zeros(K*K).reshape(K,K)
    
    for o in range(K-1,-1,-1):
        for p in range(K-1,-1,-1):
            #Lower
            if (o==0 and p==0):
                R2LoPri[o][p] = int(Fr(o,p)) ^  int (np.mod( int( (1-1.4 * (k_0/15)**2 + (k_1/15)) *precision ) ,16))
                R2HiPri[o][p] = int(Gr(o,p)) ^ int(np.mod( int(0.3 * (k_0/15) *precision)  ,16))

                G2LoPri[o][p] = int(Fg(o,p)) ^  int (np.mod( int( (1-1.4 * (k_0/15)**2 + (k_1/15)) *precision ) ,16))
                G2HiPri[o][p] = int(Gg(o,p)) ^ int(np.mod( int(0.3 * (k_0/15) *precision)  ,16))

                B2LoPri[o][p] = int(Fb(o,p)) ^  int (np.mod( int( (1-1.4 * (k_0/15)**2 + (k_1/15)) *precision ) ,16))
                B2HiPri[o][p] = int(Gb(o,p)) ^ int(np.mod( int(0.3 * (k_0/15) *precision)  ,16))
            elif (o!=0 and p==0):
                R2LoPri[o][p] = int(Fr(o,p)) ^ int(np.mod( int( (1-1.4 * (int("0b"+R2Lo[o-1][K-1],2)/15)**2 + (int("0b"+R2Hi[o-1][K-1],2)/15)) *precision ) ,16))
                R2HiPri[o][p] = int(Gr(o,p)) ^ int(np.mod( int(0.3 * (int("0b"+R2Lo[o-1][K-1],2)/15) *precision) ,16))

                G2LoPri[o][p] = int(Fg(o,p)) ^ int(np.mod( int( (1-1.4 * (int("0b"+G2Lo[o-1][K-1],2)/15)**2 + (int("0b"+G2Hi[o-1][K-1],2)/15)) *precision ) ,16))
                G2HiPri[o][p] = int(Gg(o,p)) ^ int(np.mod( int(0.3 * (int("0b"+G2Lo[o-1][K-1],2)/15) *precision) ,16))

                B2LoPri[o][p] = int(Fb(o,p)) ^ int(np.mod( int( (1-1.4 * (int("0b"+B2Lo[o-1][K-1],2)/15)**2 + (int("0b"+B2Hi[o-1][K-1],2)/15)) *precision ) ,16))
                B2HiPri[o][p] = int(Gb(o,p)) ^ int(np.mod( int(0.3 * (int("0b"+B2Lo[o-1][K-1],2)/15) *precision) ,16))
            elif (p!=0):
                R2LoPri[o][p] = int(Fr(o,p)) ^ int(np.mod( int( (1-1.4 * (int("0b"+R2Lo[o][p-1],2)/15)**2 + (int("0b"+R2Hi[o][p-1],2)/15)) *precision ) ,16))
                R2HiPri[o][p] = int(Gr(o,p)) ^ int(np.mod( int(0.3 * (int("0b"+R2Lo[o][p-1],2)/15) *precision) ,16))

                G2LoPri[o][p] = int(Fg(o,p)) ^ int(np.mod( int( (1-1.4 * (int("0b"+G2Lo[o][p-1],2)/15)**2 + (int("0b"+G2Hi[o][p-1],2)/15)) *precision ) ,16))
                G2HiPri[o][p] = int(Gg(o,p)) ^ int(np.mod( int(0.3 * (int("0b"+G2Lo[o][p-1],2)/15) *precision) ,16))

                B2LoPri[o][p] = int(Fb(o,p)) ^ int(np.mod( int( (1-1.4 * (int("0b"+B2Lo[o][p-1],2)/15)**2 + (int("0b"+B2Hi[o][p-1],2)/15)) *precision ) ,16))
                B2HiPri[o][p] = int(Gb(o,p)) ^ int(np.mod( int(0.3 * (int("0b"+B2Lo[o][p-1],2)/15) *precision) ,16))

           
                
            

    #Recombine encrypted matrices
    R2HiPri = R2HiPri.reshape(1,K*K)[0].tolist()
    R2LoPri = R2LoPri.reshape(1,K*K)[0].tolist()

    G2HiPri = G2HiPri.reshape(1,K*K)[0].tolist()
    G2LoPri = G2LoPri.reshape(1,K*K)[0].tolist()

    B2HiPri = B2HiPri.reshape(1,K*K)[0].tolist()
    B2LoPri = B2LoPri.reshape(1,K*K)[0].tolist()


    Q_2 = []

    R2, G2, B2 = [], [], []

    for q in range(len(R2HiPri)):
        value = "0b" + bin(int(R2HiPri[q]))[2:].zfill(4) + bin(int(R2LoPri[q]))[2:].zfill(4)
        R2.append(int(value,2))

        value = "0b" + bin(int(G2HiPri[q]))[2:].zfill(4) + bin(int(G2LoPri[q]))[2:].zfill(4)
        G2.append(int(value,2))

        value = "0b" + bin(int(B2HiPri[q]))[2:].zfill(4) + bin(int(B2LoPri[q]))[2:].zfill(4)
        B2.append(int(value,2))

   

    ''' Part 3.2: Step 2'''
    #Reshape array into 2D array for coordinates
    print("Retrieving decryption arrays from file...")
    loadedArray = np.load("DecryptionData/Color{}.npy".format(fileName),allow_pickle=True)

    rVals, gVals, bVals = loadedArray
    sr1,sr2,sr3,sr4,sr5 = rVals
    sg1,sg2,sg3,sg4,sg5 = gVals
    sb1,sb2,sb3,sb4,sb5 = bVals
    print("Splicing into and unscrambling virtual Rubik's cube...")

    xSub, ySub = x_2n[0:1000], y_2n[0:1000]

    S6 = binaryMask(xSub)   #0=Row rotation, 1=column rotation
    S7 = binaryMask(ySub)   #0=left/up, 1=right/down

    S8 = generateTernaryChaoticMatrices(xSub,K)
    S9 = generateTernaryChaoticMatrices(ySub,4)

    #Reverse to undo
    S6, S7, S8, S9 = list(reversed(S6)), list(reversed(S7)), list(reversed(S8)), list(reversed(S9))

    #Flip direction bits
    S7 = [l^1 for l in S7]

    sR, sG, sB = np.array(R2).reshape((K,K)), np.array(G2).reshape((K,K)), np.array(B2).reshape((K,K))

    sR, c, e, f, g, h = scrambleRubiksCube(sR,sr1,sr2,sr3,sr4,sr5,S6,S7,S8,S9)
    sG, c, e, f, g, h = scrambleRubiksCube(sG,sg1,sg2,sg3,sg4,sg5,S6,S7,S8,S9)
    sB, c, e, f, g, h = scrambleRubiksCube(sB,sb1,sb2,sb3,sb4,sb5,S6,S7,S8,S9)
   

    R2, G2, B2 = list(sR.reshape(1,K*K)[0]), list(sG.reshape(1,K*K)[0]), list(sB.reshape(1,K*K)[0])
    print("Done with Rubik's cube transformation.")
    coordOnes = []
    for a in range(K):
        for b in range(K):
            coordOnes.append( (a+1,b+1,0))

    
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

    minY = min(item[1] for item in unnormalizedSeq)
    maxY = max(item[1] for item in unnormalizedSeq)
    #Begin normalizing values
    xNorm, yNorm = [],[]

    for m in unnormalizedSeq:
        xNorm.append(   (  (m[0]-minX) * (K)    )  /  (maxX-minX)     )
        yNorm.append(   (  (m[1]-minY) * (K)    )  /  (maxY-minY)     )

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
   
    tR, tG, tB = np.array(R2), np.array(G2), np.array(B2)
    sortedIndices = np.argsort(L_primeX)
    R2, G2, B2 = tR[sortedIndices],tG[sortedIndices],tB[sortedIndices]


    R2, G2, B2 = R2.tolist(), G2.tolist(), B2.tolist()
    

    print("Saving decrypted image to file...")

    fileHeader = "P3\n# Decrypted Color Image\n{} {}\n255\n".format(K,K)

    R2 = [str(int(x))+"\n" for x in R2]
    G2 = [str(int(x))+"\n" for x in G2]
    B2 = [str(int(x))+"\n" for x in B2]
    

    Q_0 =[]

    for i in range(len(R2)):
        Q_0.append(R2[i])
        Q_0.append(G2[i])
        Q_0.append(B2[i])

    fileContent = "".join(Q_0)
    fileContent = fileHeader + fileContent

    if useDefault:
        decryptedImage = open("TestImages/ColorDecrypted{}.ppm".format(fileName),"w")
    else:
        decryptedImage = open("ColorDecrypted{}.ppm".format(fileName),"w")
    decryptedImage.write(fileContent)
    decryptedImage.close()

    print("Done.")

if __name__ == "__main__":
    main()
