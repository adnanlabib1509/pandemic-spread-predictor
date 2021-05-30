# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 02:35:24 2021

@author: Adnan Labib
"""
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, GRU
from keras.layers import Dropout
import tensorflow as tf

def deep_gru(num_seq, num_features):
    
    model = Sequential()
    model.add(GRU(units = 100, return_sequences = True, activation = "relu", 
                  input_shape = (num_seq, num_features)))
    model.add(GRU(units = 100, activation = "relu", 
                  input_shape = (num_seq, num_features)))
    model.add(Dense(units = 1))
    return model




