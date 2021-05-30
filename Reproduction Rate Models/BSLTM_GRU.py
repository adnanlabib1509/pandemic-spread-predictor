# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 18:30:01 2021

@author: Adnan Labib
"""
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Bidirectional, GRU
from keras.layers import Dropout
import tensorflow as tf

def blstm_gru(num_seq, num_features):
    
    model = Sequential()
    model.add(Bidirectional(LSTM(units = 100, activation='relu', return_sequences = True,  input_shape = (num_seq, num_features))))
    model.add(GRU(units = 100, activation = "relu", input_shape = (num_seq, num_features)))
    model.add(Dense(units = 100))
    model.add(Dense(units = 1))
    return model