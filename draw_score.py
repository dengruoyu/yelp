import numpy as np
import matplotlib.pyplot as plt
x = []

file1 = open("res_p")
score_r = []
length_r = []
score_w = []
length_w = []
'''
for line in file1:
    line = line.split(" ")
    if(line[2] == '1'):
        score_r.append(float(line[-1]))
        length_r.append(int(line[0]))
    if(line[2] == '0'):
        score_w.append(float(line[-1]))
        length_w.append(int(line[0]))
plt.scatter(length_r,score_r,s = 10,color = 'g')
plt.scatter(length_w,score_w,s = 10,color = 'r')
plt.show()
'''

predict = []
for line in file1:
    line = line.split(" ")
    tmp = float(line[3])-float(line[2])
    if((line[1]) == 'True'):
        score_r.append(tmp)
        length_r.append(int(line[0]))
    if((line[1]) == 'False'):
        score_w.append(tmp)
        length_w.append(int(line[0]))
plt.scatter(length_r,score_r,s = 10,color = 'g')
plt.scatter(length_w,score_w,s = 10,color = 'r')
plt.show()
