#!/usr/bin/env python3

#################################################
#           pastAI goes to gym class            #
#################################################

import os
import sys
import math

import numpy as np
import pandas as pd

import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



#################################################
# parameters

debug = False
id_1 = 2 # id where tastes of first ingredients are
id_2 = 3 # id where tastes of second ingredients are
max_len = 112 # length of longest array


#################################################
# load dataset

dataset = pd.read_csv('./data/ingredientCombinations.csv')


#################################################
# make data beautiful

#Changing pandas dataframe to numpy array
x = dataset.iloc[:,:4].values
y = dataset.iloc[:,4:5].values

# order x

def isNaN(num):
    return num != num




known_words = {}
word_count = 0
def get_word_id(word):
    global word_count
    try: 
        return known_words[word]
    except:
        known_words[word] = word_count
        word_count += 1
        return known_words[word]

x_ordered = []
x_max = 0

for obj in x:
    x_1 = []
    x_2 = []

    x_1.append(get_word_id(obj[0]))
    x_2.append(get_word_id(obj[1]))

    # split tastes of every ingredient
    flavours_1 = obj[id_1]
    str(flavours_1)
    if not isNaN(flavours_1):
        flavours_1 = flavours_1.split('@')
        for flavour in flavours_1:
            x_1.append(get_word_id(flavour))

    flavours_2 = obj[id_2]
    str(flavours_2)
    if not isNaN(flavours_2):
        flavours_2 = flavours_2.split('@')
        for flavour in flavours_2:
            x_2.append(get_word_id(flavour))

    # make all same length (max_len)    
    if len(x_1) > x_max:
        x_max = len(x_1)

    while(len(x_1) <= max_len):
        x_1.append(0)

    while(len(x_2) <= max_len):
        x_2.append(0)

    
    x_ordered.append(np.concatenate((x_1, x_2)))
    #x_ordered.append([x_1, x_2])


# [[1], [0,5], [1]]
# [[[z1][z2]],[[z1][z2]],[]]

if debug:
    print('x max length is {}'.format(x_max))
    print('x is: {}'.format(x_ordered))
    print('y is: {}'.format(y))

# x_ordered[0] = [[ing_1, taste_1, ..., taste_n][ing_2, taste_1, ..., taste_n]]


#################################################
# split in train and eval data

x_train,x_test,y_train,y_test = train_test_split(x_ordered,y,test_size = 0.1)

if debug:
    print(len(x_train))
    print(len(y_train)) 

#print('row: {}, column: {}'.format(x_train.shape[0], x_train.shape[1]))
#x_train = [x_train]

#################################################
# dont eat salad

model = Sequential()
model.add(Dense(256, input_dim=226, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
print(model.summary())
history = model.fit(x_train, y_train, epochs=2, batch_size=32)
#history = model.fit(x_train, y_train,validation_data = (x_test,y_test), epochs=100, batch_size=32)


#################################################
# look at result of chicken and rice
x_ordered = [x_ordered]
print(model(tf.convert_to_tensor(x_train[:1])))

# death is not the end, nn lives on (1541, william s.)