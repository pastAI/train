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

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Activation, Flatten, MaxPooling1D
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters


#################################################
# parameters

debug = False
save_lite_model = True
id_1 = 2 # id where tastes of first ingredients are
id_2 = 3 # id where tastes of second ingredients are
max_len = 150 # length of longest array TOOO LONG BUT TOO LAZY TO CHECK APROPRIATE SIZE
LOG_DIR = 'logs'


#################################################
# load dataset

dataset = pd.read_csv('./data/ingredientCombinations.csv')


#################################################
# make data beautiful

#Changing pandas dataframe to numpy array
x = dataset.iloc[:,:4].values
y = dataset.iloc[:,4:5].values
y_original = y
x_original = x

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

def order_flavour(row, pos):
    flavours = x[row][pos]
    str(flavours)
    if not isNaN(flavours):
        flavours = flavours.split('@')
        for flavour in flavours:
            flavour_id = get_word_id(flavour)
            x_table[row][flavour_id]=1

x_table = np.zeros((len(x), (max_len*2)))

for row in range(len(x)):
    order_flavour(row, id_1)
    order_flavour(row, id_2)
    # split tastes of every ingredient


sc = StandardScaler()
x_table = sc.fit_transform(x_table)

y *= 100
y = y.astype(int)

if debug:
    print('x is: {}'.format(x_table))
    print('x is: {}'.format(x_table.shape))
    print('y is: {}'.format(y))
    print('y is: {}'.format(y.shape))


#################################################
# split in train and eval data

x_train = [x_table]
y_train = y


if debug:
    print(len(x_train))
    print(len(y_train))


#################################################
# dont eat salad, become a tuna

def build_model(hp):
    model = Sequential()
    model.add(Dense(hp.Int('input_units', min_value=32, max_value=512, step=32), input_dim=(max_len * 2), activation='relu'))
    
    layer_count = np.square(hp.Int('n_layers', 1, 3))
    for i in range(layer_count):
        model.add(Dense(hp.Int(f"conv_{i}_units", min_value=32, max_value=512, step=32), activation='relu'))

    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'acc'])

    return model

tuner = RandomSearch(
    build_model,
    objective='mae',
    max_trials=5,
    executions_per_trial=1,
    directory=LOG_DIR,
    project_name='pastAI')

tuner.search(x=x_train,
             y=y_train,
             epochs=100,
             batch_size=32)

tuner.results_summary()

model = tuner.get_best_models()[0]

# watch results
results = model.predict(x_train)
'''for i in range(100):
    print('{} and {} should fit {}'.format(x_original[i][0],x_original[i][1],y_original[i]))
    print('nn calculated: {} \n'.format(results[i]))'''


#################################################
# safety first

if save_lite_model:
    # convert to tensor lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # save model
    open("models/pastAI.tflite", "wb").write(tflite_model)

# death is not the end, nn lives on (1541, william s.)