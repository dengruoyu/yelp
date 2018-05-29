import numpy
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

import csv
from math import * 
import random


csvfile1 = file('dataset/rate.csv', 'r+')
reader1 = csv.reader(csvfile1)


info = {}
for line in reader1:
    if (line[0] != "Userid"):
        if(info.has_key(line[0]) == 0):
            if(float(line[-1])<=0.5):
                info[line[0]] = [[],[]]
                info[line[0]][0] = [float(line[2])]
                #info[line[0]][1] = [line[2]]
                info[line[0]][1] = 1
            if(float(line[-1])>0.5):
                info[line[0]] = [[],[]]
                info[line[0]][0] = [float(line[2])]
                #info[line[0]][1] = [line[2]]
                info[line[0]][1] = 0
        else:
            if(float(line[-1])<=0.5):
                info[line[0]][0].append(float(line[2]))
                #info[line[0]][1].append(line[2])
                info[line[0]][1] = 1
            if(float(line[-1])>0.5):
                info[line[0]][0].append(float(line[2]))
                #info[line[0]][1].append(line[2])
                info[line[0]][1] = 0
  

data = []

for i in info:
    if(len(info[i][0]) >= 2 ):
        data.append([info[i][0],info[i][1]])

X_train = []
X_test = []
y_train = []
y_test = []
k = 0.7



random.shuffle(data)
for i in range(len(data)):
    if(i <= k * len(data)):
        X_train.append(data[i][0])
        y_train.append(data[i][1])
    else:
        X_test.append(data[i][0])
        y_test.append(data[i][1])

print len(X_train),len(y_train),len(X_test),len(y_test)

sequenceLength = 200

X_train=sequence.pad_sequences(X_train, dtype = 'float', maxlen=sequenceLength, padding='post')
X_test=sequence.pad_sequences(X_test, dtype = 'float',maxlen=sequenceLength, padding='post')
print X_train.shape[0],X_train.shape[1]

X_train = numpy.reshape(X_train, (X_train.shape[0],  X_train.shape[1],1))
X_test = numpy.reshape(X_test, (X_test.shape[0],  X_test.shape[1],1))
print X_train
model = Sequential()
model.add(Masking(mask_value=0,input_shape=(sequenceLength,1)))
model.add(LSTM(30, dropout_W=0.3, dropout_U=0.3, input_shape=(sequenceLength,1)))
#model.add(LSTM(512,dropout_W=0.3, dropout_U=0.3,return_sequences=False))
#model.add(Dense(10, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='relu'))
#adam = optimizers.Adam(lr = 0.1)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy',recall,precision,f1])
print(model.summary())


history = model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=20,batch_size=64)


scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

y_predict = model.predict(X_test)
for i in range(len(X_test)):
    length = 0
    while(length<200 and X_test[i][length] != 0 ):
        length += 1
    tmp = 0
    if (y_predict[i][0]>=0.5):
        tmp = 1
    error = y_test[i] - y_predict[i][0]

    #print length,tmp == y_test[i],y_test[i],error
#print(history.history)