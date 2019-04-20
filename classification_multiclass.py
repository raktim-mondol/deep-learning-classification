#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 18:44:50 2018
@author: raktim
"""
#import necessary library
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dropout
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score, auc, roc_curve, cohen_kappa_score
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#fix random seed for reproducibility
seed = 745
numpy.random.seed(seed)

# function that convert the boolean label given, y=[[True, False, False], [False, True, False],[False, False, True]]  
# into integer label y=1,2,3
# But if you your data has more than three label then 

def bool_to_int(label):
    import numpy as np
    label=label.astype(int)
    row=label.shape[0]
    col=label.shape[1]
    
    label_pred=[]
    val_get=0;
    for i in range(row):
        store='';
        for j in range(col):
            name=str(label[i][j])
            store=store+name

        if (store =='100'):
            val_get=0;
           
        elif (store=='010'):
            val_get=1;
        
        elif (store=='001'):
            val_get=2;
        #if you have more than three label then change this accordingly
        #such as if you have y=0,1,2,3 (four classes) then 
        #it will be '1000' where val_get=0, '0100' where val_get=1
        #'0010' where val_get=2, '0001' where val_get=3
        label_pred.append(val_get)
    
    y_final=np.array(label_pred)
    return y_final

# load dataset
iris = datasets.load_iris()
X = iris.data[:,0:4]  #main data
Y = iris.target  #label

# label encoder necessary when label in the data are  
# wrtitten in word
# or label in the data are not organized or not starting from zero 
# such as given y=[2, 2, 3, 4, 4, 4] then after encoding it will be y=[0, 0, 1, 2, 2, 2] 
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = seed) #spilt the dataset into 70% train, 30% test data.

y_test_int=y_test
y_train_binarize=label_binarize(y_train,classes=[0,1,2])
y_test_binarize=label_binarize(y_test,classes=[0,1,2])
#alternatively use the following line
#same as label binarizer
#y_one_hot_encoded = np_utils.to_categorical(Y) 

# Initialising the ANN
clf = Sequential()
       
# Adding the input layer and the first hidden layer
clf.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4))
clf.add(Dropout(p = 0.01))
clf.add(Dense(units = 50, activation='relu'))
#clf.add(Dense(units = 10, activation='relu'))
#you can increase the layer by uncommenting the above line 

# Adding the output layer
clf.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax'))
#change activation into sigmoid if binary class is used

# Compiling the ANN
clf.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#for binary classification, loss will be binary crossentropy

#fit the model
clf.fit(x_train, y_train_binarize, batch_size = 20, 
        epochs = 30, verbose=1, 
        validation_data=(x_test, y_test_binarize)) # for ANN keras

clf_score = clf.evaluate(x_test, y_test_binarize, verbose=1, batch_size=20)   #for ANN(using KERAS)
y_pred_proba=clf.predict(x_test) #print as probability score such as y=[0.1, 0.8, 0.1]
y_pred_bool=y_pred_proba>0.5 #convert into boolean
y_pred_int=bool_to_int(y_pred_bool) #label converted into integer so that below performance metrics do work. 

#print all the performance metrics

print("Test Accuracy: \n%s: %.2f%%" % (clf.metrics_names[1], clf_score[1]*100))
#metrics_names[1] means accuracy metrics_names[0] means loss
print 'Train Test AUC Score', roc_auc_score(y_test_binarize, y_pred_proba, average='macro', sample_weight=None)
print('Precision Weighted', precision_score(y_test_int, y_pred_int, average='weighted'))
print('MCC Score:', matthews_corrcoef(y_test_int,y_pred_int))
print('Recall Macro:', recall_score(y_test_int, y_pred_int, average='weighted'))
print('F1 Micro :', f1_score(y_test_int, y_pred_int,average='weighted'))
print('Kappa Score:', cohen_kappa_score(y_test_int, y_pred_int, labels=None, weights=None, sample_weight=None))


