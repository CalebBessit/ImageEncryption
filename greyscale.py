m = 270
n = 270

newlines = []

fileNames = ["","Explosion", "Fence","Ishigami","Pikachu","PowerLines","Shirogane","Tower"]

for f in range(1,8):
    fileName= fileNames[f]
    newlines = []
    file = open("TestImages/{}.ppm".format(fileName),"r")
    newFile = open("TestImages/Grey{}.ppm".format(fileName),"w")
    count = 0
    runningTotal =0

    lines = file.readlines()

    newlines.append(lines[0].replace("3","2"))

    for a in range(3):
        newlines.append(lines[a+1])


    m, n = lines[2].replace("\n","").split(" ")
    K = max(int(m), int(n))

    count = 0
    runningTotal =0
    for b in range(4,len(lines)):
        line = lines[b]
        line = line.replace("\n","")
        count +=1
        runningTotal+=int(line)
        if count%3==0:
            value = int(runningTotal/3)
            runningTotal=0
            newlines.append(str(value) + "\n")


    string=""
    file.close()
    string = "".join(newlines)
    # print(len(newlines))
    # print(newlines)
    start=len(string)-20
    # print(string[start:]+"test")

    print("largest: ",K)
    print("done " + fileName)
    newFile.write(string)
    newFile.close()
    

    
    


