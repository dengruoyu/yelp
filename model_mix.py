f1 = open('dataset/lstm.txt','r')
f2 = open('dataset/ada.txt','r')
f3 = open('dataset/lstm_tmp.txt','r')
f4 = open('dataset/ada_tmp.txt','r')
train_l = []
train_a = []
train_v = []

test_l = []
test_a = []
test_v = []

for line in f1:
    line = line.split(" ")
    train_l.append(float(line[-1]))

for line in f2:
    line = line.split(" ")
    train_a.append([float(line[0]),float(line[2])])
    train_v.append(int(line[-1]))

for line in f3:
    line = line.split(" ")
    test_l.append(float(line[-1]))

for line in f4:
    line = line.split(" ")
    test_a.append([float(line[0]),float(line[2])])
    test_v.append(int(line[-1]))

data_train = []
for i in range(len(train_a)):
    tmp = [train_l[i],train_a[i][0],train_a[i][1],train_v[i]]
    data_train.append(tmp)
data_test = []
for i in range(len(test_a)):
    tmp = [test_l[i],test_a[i][0],test_a[i][1],test_v[i]]
    data_test.append(tmp)

from sklearn.cross_validation import train_test_split 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import random
from sklearn import metrics
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC,NuSVC,LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
random.shuffle(data_test)
random.shuffle(data_train)

train_X = []
train_y = []
test_X = []
test_y = []


for i in data_train:
    train_X.append([i[0],i[1],i[2]])
    train_y.append(i[3])
for i in data_test:
    test_X.append([i[0],i[1],i[2]])
    test_y.append(i[3])

from sklearn import preprocessing

train_X = preprocessing.scale(train_X)

test_X = preprocessing.scale(test_X)

#clf = GradientBoostingClassifier(max_depth=7,subsample=0.9,loss='exponential')
clf = NuSVC(nu=0.7,random_state=True,kernel = 'sigmoid',coef0=0.13,decision_function_shape='ovo')

clf.fit(train_X, train_y)
y_pred = clf.predict(test_X)
print metrics.accuracy_score(test_y, y_pred)
print metrics.precision_score(test_y, y_pred)
print metrics.recall_score(test_y, y_pred)
print metrics.f1_score(test_y, y_pred)


import pickle 
with open('model/mix.picksle', 'wb') as f:
    pickle.dump(clf, f)