#Program which adds noise to an encrypted image to test decryption
#Caleb Bessit
#12 October 2023

fileName = "Pikachu"

file = open("TestImages/GreyEncrypted{}.ppm".format(fileName),"r")

modes = ["30Fill","60Fill"]

lines   = file.readlines()
header  = lines[0:4]
lines   = lines[4:len(lines)]



#Calculate proportion of images to be filled
num30 = int( (1/3) * len(lines))
num60 = int( 0.6* len(lines))

fill30, fill60, gauss30 = [], [], []

for i in range(len(lines)):
    lines[i] = int(lines[i].replace("\n",""))

    if i<num30:
        fill30.append("255\n")
    else:
        fill30.append(str(lines[i])+"\n")

    if i<num60:
        fill60.append("0\n")
    else:
        fill60.append(str(lines[i])+ "\n")

file.close()


fill30 = "".join(header) + "".join(fill30)
fill60 = "".join(header) + "".join(fill60)

file = open("TestImages/GreyEncrypted{}30Filled.ppm".format(fileName),"w")
file.write(fill30)
file.close()

print("Done with 30% file.")

file = open("TestImages/GreyEncrypted{}60Filled.ppm".format(fileName),"w")
file.write(fill60)
file.close()

print("Done with 60% file.")


    

    

