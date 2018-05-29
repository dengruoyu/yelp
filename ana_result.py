import math

def cal_4(a,b,c,d):
    x1 = float((a+c))/(a+b+c+d)
    x2 = float(a)/(a+d)
    x3 = float(a)/(a+b)
    x4 = 2*x2*x3/(x2+x3)
    return x1,x2,x3,x4

f = open('dataset/ada_tmp.txt','r')
good_r = 0
good_w = 0
bad_r = 0
bad_w = 0
for line in f:
    line = line.split(" ")
    if(int(line[3]) == 0):
        if(line[1] == 'True'):
            bad_r += 1
        if(line[1] == 'False'):
            bad_w += 1
    if(int(line[3]) == 1):
        if(line[1] == 'True'):
            good_r += 1
        if(line[1] == 'False'):
            good_w += 1
    #tmp = float(line[3])
    #print math.exp(tmp)/(math.exp(tmp)+math.exp(1-tmp))

print good_r,good_w,bad_r,bad_w
print cal_4(good_r,good_w,bad_r,bad_w)

f = open('dataset/lstm_tmp.txt','r')
good_r = 0
good_w = 0
bad_r = 0
bad_w = 0
for line in f:
    line = line.split(" ")
    if(int(line[2]) == 0):
        if(line[1] == 'True'):
            bad_r += 1
        if(line[1] == 'False'):
            bad_w += 1
    if(int(line[2]) == 1):
        if(line[1] == 'True'):
            good_r += 1
        if(line[1] == 'False'):
            good_w += 1
    #tmp = float(line[3])
    #print math.exp(tmp)/(math.exp(tmp)+math.exp(1-tmp))


print good_r,good_w,bad_r,bad_w
print cal_4(good_r,good_w,bad_r,bad_w)
f = open('res','r')
good_r = 0
good_w = 0
bad_r = 0
bad_w = 0
for line in f:
    line = line.split(" ")
    if(int(line[1]) == 0):
        if(line[0][1] == '0'):
            bad_r += 1
        if(line[0][1] == '1'):
            bad_w += 1
    if(int(line[1]) == 1):
        if(line[0][1] == '1'):
            good_r += 1
        if(line[0][1] == '0'):
            good_w += 1
    #tmp = float(line[3])
    #print math.exp(tmp)/(math.exp(tmp)+math.exp(1-tmp))

print good_r,good_w,bad_r,bad_w
print cal_4(good_r,good_w,bad_r,bad_w)

