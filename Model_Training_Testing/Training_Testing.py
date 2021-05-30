# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 22:49:02 2021

@author: Adnan Labib
"""
# Importing packages
import math
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed
from keras.layers import Dropout
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from LSTM import lstm
from Deep_LSTM import deep_lstm
from Bi_LSTM import bi_lstm
from Deep_LSTM_2 import deep_lstm_2
from GRU import gru
from Deep_GRU import deep_gru
from CNN_LSTM import cnn_lstm
from Conv_LSTM import conv_lstm

###########################################################
# Preprocessing
###########################################################

# Read data into a dataframe from the csv file
df = pd.read_csv("Current_Dataset.csv")

# convert all NaN values to zero
df = df.fillna(0)


all_countries = ["United States", "India", "United Kingdom", "Germany",
                 "Argentina", "South Africa", "Iraq", "Bangladesh",
                 "Malaysia", "Tunisia"]

for country in range(len(all_countries)):
    # Select a country
    world_df=df[(df.location==all_countries[country])]
 
    
       
    # Select features
    new_cases = world_df[["new_cases"]].values.tolist()
    #print(all_countries[i])
    
    # Save a copy of unscaled data 
    unscaled_new_cases = new_cases
    
    # Reshape the data
    new_cases = np.array(new_cases).reshape(-1,1)

    x_train=[]
    y_train = []
    
    
    
    for i in range(292):
        for j in range(10):
            x_train.append(new_cases[i+j])
        #for k in range(7):
        y_train.append(new_cases[i+10,0])
    
    #Convert to numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    # Reshape the data
    x_train = x_train.reshape((292,10,1))
    
   
    # Initialize the model here
    model = bi_lstm(10,1)

    # Compile and fit the model
    model.compile(optimizer = "Adam", loss = 'mean_squared_error') 
    model.fit(x_train, y_train, epochs = 250, batch_size = 10, verbose = 0)
    
    
    ###########################################################
    # Testing
    ###########################################################
     
    # # Getting the testing data
    x_test=[]
    y_test = []
    for i in range(293,366):
        for j in range(10):
            x_test.append(new_cases[i+j])
        y_test.append(new_cases[i+10,0])
     
   
    # Convert to numpy array
    x_test, y_test = np.array(x_test), np.array(y_test)
    
    # Reshape the data
    x_test = x_test.reshape((73,10,1))
    #y_test = y_test.reshape((73,7))
    
    # Make the predictions for the testing data
    predictions = model.predict(x_test)
    

        
    # Get the Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, predictions)
    print("Country: ", all_countries[country], " MAE: ", mae)

   

    
    
    
    
    
    
    
    
    
    