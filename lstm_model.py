import numpy
import random
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import Dense, Input, Flatten, Dropout
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall   

def precision(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision 

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding,Masking
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
def get_lstm_model(good_train,good_test,bad_train,bad_test):

    train = []
    test = []
    
    X_train = []
    X_test = []
    
    y_train = []
    y_test = []


    for i in good_train:
        train.append([i[0],1])
    for i in bad_train:
        train.append([i[0],0])

    random.shuffle(train)
    for i in train:
        X_train.append(i[0])
        y_train.append(i[1])

    
    for i in good_test:
        test.append([i[0],1])
    for i in bad_test:
        test.append([i[0],0])

    for i in test:
        X_test.append(i[0])
        y_test.append(i[1])
    
    print len(X_train),len(y_train),len(X_test),len(y_test)

    sequenceLength = 200

    X_train=sequence.pad_sequences(X_train, dtype = 'float', maxlen=sequenceLength, padding='post')
    X_test=sequence.pad_sequences(X_test, dtype = 'float',maxlen=sequenceLength, padding='post')

    X_train = numpy.reshape(X_train, (X_train.shape[0],  X_train.shape[1],1))
    X_test = numpy.reshape(X_test, (X_test.shape[0],  X_test.shape[1],1))

    model = Sequential()
    model.add(Masking(mask_value=0,input_shape=(sequenceLength,1)))
    model.add(LSTM(15, dropout_W=0.5, dropout_U=0.5, input_shape=(sequenceLength,1)))
    #model.add(LSTM(512,dropout_W=0.3, dropout_U=0.3,return_sequences=False))
    #model.add(Dense(10, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    #adam = optimizers.Adam(lr = 0.1)
    model.compile(loss='msle', optimizer='adam', metrics=['accuracy',recall,precision,f1])
    print(model.summary())

    f = open("dataset/lstm_tmp.txt",'w')
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=5,batch_size=64)


    scores = model.evaluate(X_test, y_test, verbose=0)
    print ("Accuracy: %.2f%%" % (scores[1]*100))
 

    y_predict = model.predict_proba(X_test)
    
    for i in range(len(X_test)):
        length = 0
        while(length<200 and X_test[i][length] != 0 ):
            length += 1
        tmp = 0
        if (y_predict[i][0]>=0.5):
            tmp = 1
        sentence = ""
        sentence = str(length)+" "+str(tmp == y_test[i])+" "+str(y_test[i])+" "+str(y_predict[i][0])+"\n"
        f.write(sentence)
    f.close()
    print(history.history)
    return y_predict