#!/usr/bin/env python3

#################################################
#           pastAI goes to gym class            #
#################################################


import os
import sys
import math

import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split


#################################################
# parameters

debug = False
id_1 = 2 # id where tastes of first ingredients are
id_2 = 3 # id where tastes of second ingredients are
max_len = 113 # length of longest array


#################################################
# load dataset

dataset = pd.read_csv('./data/ingredientCombinations.csv')


#################################################
# make data beautiful

#Changing pandas dataframe to numpy array
x = dataset.iloc[:,:4].values
y= dataset.iloc[:,4:5].values

# order x

def isNaN(num):
    return num != num


x_ordered = []
x_max = 0

for i in range(len(x)):
    x_1 = []
    x_2 = []

    x_1.append(x[i][0])
    x_2.append(x[i][1])

    # split tastes of every ingredient
    e_1 = x[i][id_1]
    str(e_1)
    if not isNaN(e_1):
        e_1 = e_1.split('@')
        for e in e_1:
            x_1.append(e)

    e_2 = x[i][id_2]
    str(e_2)
    if not isNaN(e_2):
        e_2 = e_2.split('@')
        for e in e_2:
            x_2.append(e)

    # make all same length (max_len)    
    if len(x_1) > x_max:
        x_max = len(x_1)

    while(len(x_1) < max_len):
        x_1.append(0)

    while(len(x_2) < max_len):
        x_2.append(0)


    x_ordered.append([x_1, x_2])

if debug:
    print('x max length is {}'.format(x_max))
    print('x is: {}'.format(x_ordered))
    print('y is: {}'.format(y))

# x_ordered[0] = [[ing_1, taste_1, ..., taste_n][ing_2, taste_1, ..., taste_n]]


#################################################
# split in train and eval data

x_train,x_test,y_train,y_test = train_test_split(x_ordered,y,test_size = 0.1)


#################################################
# dont eat salad

model = Sequential()
model.add(Dense(16, input_dim=113, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train,validation_data = (x_test,y_test), epochs=100, batch_size=64)


#################################################
# look at result of chicken and rice



# death is not the end, nn lives on (1541, william s.)