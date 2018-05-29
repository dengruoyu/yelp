import numpy as np
import matplotlib.pyplot as plt
x = []

file1 = open("res_a")
ada0 = []
ada1 = []

for i in range(200):
    ada0.append(0)
    ada1.append(0)
    x.append(i+1)
for line in file1:
    line = line.split(" ")
    if(int(line[2]) == 1):
        continue
    #print line[0],line[1],line[2]
    if(int(line[0]) < 200):
        if (line[1] == "True"):
            ada0[int(line[0])-1] += 1
        else:
            ada1[int(line[0])-1] += 1
    if(int(line[0]) >= 200):
        if (line[1] == "True"):
            ada0[199] += 1
        else:
            ada1[199] += 1

file2 = open("res_l")
lstm0 = []
lstm1 = []
for i in range(200):
    lstm0.append(0)
    lstm1.append(1)

for line in file2:
    line = line.split(" ")
    if(int(line[2]) == 1):
       continue
    #print line[0],line[1],line[2]
    if(int(line[0]) < 200):
        if (line[1] == "True"):
            lstm0[int(line[0])-1] += 1
        else:
            lstm1[int(line[0])-1] += 1
    if(int(line[0]) >= 200):
        if (line[1] == "True"):
            lstm0[199] += 1
        else:
            lstm1[199] += 1

'''
for i in range(1,20):
    if(ada0[i]+ada1[i]!=0):
        if(lstm0[i]+lstm1[i]!=0):
            print float(ada0[i])/(ada0[i]+ada1[i]),",",float(lstm0[i])/(lstm0[i]+lstm1[i])
        else:
            print float(ada0[i])/(ada0[i]+ada1[i]),",","no"
    else:
        if(lstm0[i]+lstm1[i]!=0):
            print "no",",",float(lstm0[i])/(lstm0[i]+lstm1[i])
        else:
            print "no",",","no"
'''

ada_rate = []
lstm_rate = []
x = []
for i in range(1,20):
    if(ada0[i]+ada1[i]!=0):
        ada_rate.append(float(ada0[i])/(ada0[i]+ada1[i]))
    if(ada0[i]+ada1[i]==0 and i>=10):
        ada_rate.append(ada_rate[i-2])
    if(lstm0[i]+lstm1[i]!=0):
        lstm_rate.append(float(lstm0[i])/(lstm0[i]+lstm1[i]))

    x.append(i+1)
plt.plot(x,ada_rate)
plt.plot(x,lstm_rate)


plt.show()
'''
y00 = np.array(lstm0)
y01 = np.array(lstm1)
plt.bar(x, y00, color='green', label='true')
plt.bar(x, y01, bottom=y00, color='red', label='wrong')
plt.legend(loc=[1, 0])
plt.show()
'''