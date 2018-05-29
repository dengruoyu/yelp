import csv
from math import * 
import random

def build_dictionary(reader1,reader2):

    info_g = {}
    info_b = {}
    #radius
    for line in reader1:
        if (line[0] != "Userid"):
            if(float(line[-1])<=0.5):
                if(info_g.has_key(line[0]) == 0):
                    info_g[line[0]] = [[],[]]
                    info_g[line[0]][0] = [float(line[3])]
                else:
                    info_g[line[0]][0].append(float(line[3]))
                    #info[line[0]][1].append(line[2])
            else:
                if(info_b.has_key(line[0]) == 0):
                    info_b[line[0]] = [[],[]]
                    info_b[line[0]][0] = [float(line[3])]
                else:
                    info_b[line[0]][0].append(float(line[3]))
                    #info[line[0]][1].append(line[2])

    
    for line in reader2:
        if(info_g.has_key(line[0])):
            for i in range(1,11):
                info_g[line[0]][1].append(int(line[i]))
        if(info_b.has_key(line[0])):
            for i in range(1,11):
                info_b[line[0]][1].append(int(line[i]))

    return info_g,info_b

def filter(info_g,info_b,threshold = 5):
    '''
        filter user whose review number is less than threshold
        and empty user in dictionary. Stored in two lists, good and bad.
    '''
    good = []
    bad = []
    for i in info_g:
        if(len(info_g[i][0]) >= threshold and len(info_g[i][1]) != []):
            tmp = [info_g[i][0],info_g[i][1]]
            good.append(tmp)
    for i in info_b:
        if(len(info_b[i][0]) >= threshold and len(info_b[i][1]) != []):
            tmp = [info_b[i][0],info_b[i][1]]
            bad.append(tmp)
    random.shuffle(good)
    random.shuffle(bad)
    return good,bad

def get_list():
    '''
        read data from csv files 
    '''
    csvfile1 = file('dataset/radius_r.csv', 'r+')
    reader1 = csv.reader(csvfile1)
    csvfile2 = file('dataset/reviewer_account.csv', 'r+')
    reader2 = csv.reader(csvfile2)


    info_g,info_b = build_dictionary(reader1,reader2)
    #print len(info_g),len(info_b)
    #1519 887

    good,bad = filter(info_g,info_b,threshold = 2)
    #print len(good),len(bad)
    #1246,429
    rate = 0.7
    good_train = good[:int(len(good)*rate)]
    good_test = good[int(len(good)*rate):]
    bad_train = bad[:int(len(bad)*rate)]
    bad_test = bad[int(len(bad)*rate):]
   
    return good_train,good_test,bad_train,bad_test

#g,b = get_list()
