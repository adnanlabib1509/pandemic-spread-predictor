# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 22:29:42 2021

@author: Adnan Labib
"""
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, GRU
from keras.layers import Dropout
import tensorflow as tf

def lstm(num_seq,num_features):
    model = Sequential()
    model.add(LSTM(units = 200, activation= 'relu', 
                   input_shape = (num_seq, num_features)))
    model.add(Dropout(0.2))
    model.add(Dense(units = 100))
    model.add(Dense(units = 100))
    model.add(Dense(units = 1))
    return model







