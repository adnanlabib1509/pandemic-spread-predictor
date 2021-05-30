# -*- coding: utf-8 -*-
"""
Created on Sun May  9 00:28:58 2021

@author: Adnan Labib
"""
# Import Required Libraries
import numpy as np
from tensorflow.keras.models import load_model, clone_model
from api_getter import get_cases
import glob
import datetime
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Bidirectional, GRU
from keras.layers import Dropout
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

###################################################################
# FUNCTION used to generate a new LSTM model
# INPUT: num_layers: the number of additional lstm layers, by default it has 1 LSTM layer
# OUTPUT: the new LSTM model
####################################################################
def generate_lstm(num_layers):
    model = Sequential()
    
    for layers in range(num_layers):
        model.add(LSTM(units = 200, activation= 'relu', return_sequences=True, input_shape = (10, 1)))
    model.add(LSTM(units = 200, activation= 'relu', input_shape = (10, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(units = 100))
    model.add(Dense(units = 100))
    model.add(Dense(units = 1))
    return model

###################################################################
# FUNCTION used to generate a new Bi-LSTM model
# INPUT: num_layers: the number of additional bi-lstm layers, by default it has 1 Bi-LSTM layer
# OUTPUT: the new Bi-LSTM model
####################################################################
def generate_bi_lstm(num_layers):
    
    model = Sequential()
    for layers in range(num_layers):
        model.add(Bidirectional(LSTM(units = 100, activation='relu', return_sequences=True, input_shape = (10, 1))))
    model.add(Bidirectional(LSTM(units = 100, activation='relu', input_shape = (10, 1))))
    model.add(Dense(units = 1))
    return model

###################################################################
# FUNCTION used to generate a new GRU model
# INPUT: num_layers: the number of additional gru layers, by default it has 1 GRU layer
# OUTPUT: the new GRU model
####################################################################
def generate_gru(num_layers):
    
    model = Sequential()
    for layers in range(num_layers):
        model.add(GRU(units = 100, return_sequences = True, activation = "relu", input_shape = (10, 1)))
    model.add(GRU(units = 100, activation = "relu", input_shape = (10,1)))
    model.add(Dense(units = 1))
    return model

###################################################################
# Function used to train the chosen model with new data
# INPUT: country: the country for which past Covid-19 cases is needed
#        model_id: specifies whether it is LSTM, BI-LSTM or GRU
#        fl_value: 1 if incremental training, 2 if new model generation
# OUTPUT: the newly trained model
###################################################################    

def training(country, model_id, fl_value, num_layers=None):
    
    if model_id==1:
        #print("LSTM Model is used")
        model_name = "vanilla_lstm"
    elif model_id==2:
        #print("Deep GRU Model is used")
        model_name = "deep_gru"
    else:
        #print("Bi-LSTM Model is used")
        model_name = "bi_lstm"
    flag = False
    best_model = None
    best_mae = float("inf")
    for filename in glob.glob('*.h5'):
        if model_name in filename:
            model = load_model(filename)
    
            initial_model = model
            
            if fl_value == 1:
                print("FL 1.0")
                new_model = clone_model(model)
                new_model.set_weights(model.get_weights())
            else:
                time = datetime.now()
                time = time.strftime("%y%d%H%M%S")
                if model_id==1:
                    #print("LSTM Model is used")
                    new_model = generate_lstm(num_layers)
                elif model_id==2:
                    #print("Deep GRU Model is used")
                    new_model = generate_gru(num_layers)
                else:
                    #print("Bi-LSTM Model is used")
                    new_model = generate_bi_lstm(num_layers)
                print("FL 2.0")
            
            
            new_cases = get_cases(country, 50)
            
            #print(new_cases)
            
            new_cases = np.array(new_cases).reshape(-1,1)
            
            x_train=[]
            y_train = []
            
            for i in range(30):
                for j in range(10):
                    x_train.append(new_cases[i+j])
                #for k in range(7):
                y_train.append(new_cases[i+10,0])
                
            #Convert to numpy arrays
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            
            # Reshape the data
            x_train = x_train.reshape((30,10,1))
            
            if fl_value==2:
                x_train = tf.cast(x_train, tf.float32)
                y_train = tf.cast(y_train, tf.float32)
            new_model.compile(optimizer = "Adam", loss = 'mean_absolute_error')  
            new_model.fit(x_train, y_train, epochs = 250, batch_size = 10, verbose = 0)
            
            # Getting the testing data
            x_test=[]
            y_test = []
            for i in range(30,40):
                for j in range(10):
                    x_test.append(new_cases[i+j])
                #for k in range(7):
                y_test.append(new_cases[i+10,0])
                
            # Convert to numpy array
            x_test, y_test = np.array(x_test), np.array(y_test)
            
            # Reshape the data
            x_test = x_test.reshape((10,10,1))
            #y_test = y_test.reshape((73,7))
            
            # Make the predictions for the testing data
            predictions_old = initial_model.predict(x_test)
            predictions_new = new_model.predict(x_test)
        
            
            mae_old = mean_absolute_error(y_test, predictions_old)
            mae_new = mean_absolute_error(y_test, predictions_new)
            print(mae_old)
            print(mae_new)
            
            if mae_new<(mae_old*0.7):
                print("New Model is used")
                #if fl_value == 2:
                    #new_model.save(str(time)+"_"+filename)
                if mae_new<best_mae:
                    
                    best_model = None
                    best_model = clone_model(new_model)
                    best_model.set_weights(new_model.get_weights())
                    best_mae = mae_new
                    if fl_value == 2:
                        flag = True
                #return new_model
            else:
                print("Old Model is used")
                #return initial_model
                if mae_old<best_mae:
                    print("REACHED HERE")
                    best_mae=mae_old
                    best_model = None
                    best_model = clone_model(initial_model)
                    best_model.set_weights(initial_model.get_weights())
                    flag = False
                    
    if flag == True:
        new_model.save(str(time)+"_"+filename)
    return best_model
                
                
    















