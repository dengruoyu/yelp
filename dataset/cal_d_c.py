f1 = open('dataset/ada_tmp.txt','r')
f2 = open('dataset/lstm_tmp.txt','r')
list1 = []
list2 = []
for line in f1:
    line = line.split(" ")
    if(float(line[-2])<=0.5):
        list1.append(0)
    if(float(line[-2])>0.5):
        list1.append(1)
for line in f2:
    line = line.split(" ")
    if(float(line[-1])<=0.5):
        list2.append(0)
    if(float(line[-1])>0.5):
        list2.append(1)

a=0
b=0
c=0
d=0
for i in range(len(list1)):
    if(list1[i] == 1 and list2[i] == 1):
        a+=1
    if(list1[i] == 0 and list2[i] == 1):
        b+=1
    if(list1[i] == 1 and list2[i] == 0):
        c+=1
    if(list1[i] == 0 and list2[i] == 0):
        d+=1

print a,b,c,d
print (b+c)*1.0/(a+b+c+d)
t = (a+b)*(a+c)*(c+d)*(b+d)
import math
q = math.sqrt(t)

print (a*d-b*c)*1.0/q
