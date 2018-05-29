from read_data import get_list
from sklearn.cross_validation import train_test_split 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import random
from sklearn import metrics

def get_ada_model(good_train,good_test,bad_train,bad_test):
    '''
        get model by user account information
    '''

    train = []
    train_X = []
    train_y = []
    test_X = []
    test_y = []

    for i in good_train:
        train.append([i[1],1])
    for i in bad_train:
        train.append([i[1],0])
    random.shuffle(train)
    for i in train:
        train_X.append(i[0])
        train_y.append(i[1])
    for i in good_test:
        test_X.append(i[1])
        test_y.append(1)
    for i in bad_test:
        test_X.append(i[1])
        test_y.append(0)

    clf = GradientBoostingClassifier(max_depth=5,subsample=0.8,loss='exponential')
    clf.fit(train_X, train_y)
    f = open("dataset/ada_tmp.txt",'w')

    predict_y = clf.predict(test_X)
    y_prob = clf.predict_proba(test_X)

    for i in range(len(test_X)):
        sentence = ""
        sentence = str(test_X[i][-1])+" "+str(test_y[i] == predict_y[i])+" "+str(y_prob[i][1])+" "+str(test_y[i])+"\n"
        f.write(sentence) 
    f.close()
    #print "total accuracy rate for account-adaboost model:",clf.score(test_X, test_y)


    '''
    for i in range(len(test_X)): 
        print int(clf.predict(test_X[i]))==test_y[i],test_y[i],test_X[i][1]
    '''
    #y_prob = clf.predict_proba(test_X)
    #for i in range(len(test_X)):
    #    print test_X[i][-1],test_y[i] == predict_y[i],y_prob[i][1], test_y[i]
    return y_prob,test_y



            

    

    
    
    


    