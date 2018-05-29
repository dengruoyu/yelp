from read_data import get_list
from adaboost_model import get_ada_model
from lstm_model import get_lstm_model
import pickle

def main():
    good_train,good_test,bad_train,bad_test = get_list()
    y2,y = get_ada_model(good_train,good_test,bad_train,bad_test)

    
    y1 = get_lstm_model(good_train,good_test,bad_train,bad_test)

    #print len(y),len(y1),len(y2)
    #print y1
    #print y2
    '''
    with open('model/mix.pickle', 'rb') as f:
        clf2 = pickle.load(f)

        for i in range(len(y1)):
            tmp = [y1[i][0],y2[i][1]]
            #print tmp
            res = clf2.predict(tmp)
            print res,y[i]
    '''
main()