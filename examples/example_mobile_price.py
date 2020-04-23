#!/usr/bin/env python3

#Dependencies
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import keras
from keras.models import Sequential
from keras.layers import Dense

#dataset import
dataset = pd.read_csv('./data/train.csv')
#dataset.head(10) #Return 10 rows of data

#Changing pandas dataframe to numpy array
X = dataset.iloc[:,:20].values
y = dataset.iloc[:,20:21].values

#Normalizing the data
sc = StandardScaler()
X = sc.fit_transform(X)

ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)

# Neural network
model = Sequential()
model.add(Dense(16, input_dim=20, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#history = model.fit(X_train, y_train, epochs=100, batch_size=64)
history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=100, batch_size=64)


'''
#check models performance on test data
y_pred = model.predict(X_test)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))

a = accuracy_score(pred,test)
print('Accuracy is:', a*100)
'''