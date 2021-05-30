# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 13:10:52 2021

@author: Adnan Labib
"""
# Importing Required libraries
from keras.models import load_model
import math
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Bidirectional
from keras.layers import Dropout, GRU
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import math

# Loading all the models
model_lstm = load_model('model_lstm.h5')
model_deep_gru = load_model('model_deep_gru.h5')
model_bi_lstm = load_model('model_bi_lstm.h5')

# Read data into a dataframe from the csv file
df = pd.read_csv("Current_Dataset.csv")

# convert all NaN values to zero
df = df.fillna(0)


model_select = [1,2,3,2,3,3,3,3,1,3,3,1,3,2,2,1,3,2,3,2,2,1,2,2,3,1,1,3,3,3,2,2,3,2,3,3,3,1,3,2,3,2,2,2,3,3,3,2,3,2,1,2,2,3,2,3,2,2,3,3,3,1,2,3,3,3,3,2,3,3,3,3,3,3,2,3,2,2,3,2,3,2,2,3,3,3,2,2,3,3,3,2,3,1,1,2,3,1,2,3,3,3,3,3,3,3,3,2,3,3,3,3,2,2,2,3,2,3,2,3,2,3,1,3,3,2,1,1,3,2,3,2,2,3,2,3,2,3,2,3]

all_countries = df.location.unique()
mae_list=[]
for country in range(len(all_countries)):

    # Getting the testing data
    x_test=[]
    y_test = []

    test_df=df[(df.location==all_countries[country])]
    test_cases = test_df[["new_cases", "stringency_index"]].values.tolist()
    test_cases = np.array(test_cases).reshape(-1,2)
    # value = 293, 366
    for i in range(293,366):
        for j in range(10):
            x_test.append(test_cases[i+j,0])
        y_test.append(test_cases[i+10,1])

    # =============================================================================
    # # Normalize the testing data
    # x_test = x_scaler.fit_transform(x_test)
    # =============================================================================

    # Convert to numpy array
    x_test, y_test = np.array(x_test), np.array(y_test)

    # value = 73
    # Reshape the data
    x_test = x_test.reshape((73,10,1))
    
    if model_select[country]==1:
        model = model_lstm
    elif model_select[country]==2:
        model = model_deep_gru
    else:
        model = model_bi_lstm
    
    # Make the predictions for the testing data
    predictions = model.predict(x_test)

    # =============================================================================
    # # De-Normalize
    # #predictions = predictions.reshape(-1,1)
    # predictions = scaler.inverse_transform(predictions)
    # #predictions = predictions.reshape(-1)
    # =============================================================================
    
    for item in range(len(predictions)):
        if predictions[item]<0:
            predictions[item] = 0
    
# =============================================================================
#     # Get the Mean Absolute Error (MAE)
#     mae = mean_absolute_error(y_test, predictions)
#     print(all_countries[country], " MAE: ", mae)
# =============================================================================
    mse = mean_squared_error(y_test,predictions)
    rmse = math.sqrt(mse)
    print(all_countries[country], " MAE: ", rmse)

    #mae_list.append(mae)
    
    #predictions = predictions.reshape(-1)
    #print(predictions.tolist()[0])

#print(sum(mae_list)/len(mae_list))

    
# =============================================================================
#     # Get the graph showing the comparison between predicted and actual 
#     x_range=[]
#     y_range=[]
#     test_date = test_df[["date"]].values.tolist()
#             
#     for i in range(303,376):
#         x_range.append(i)
#         y_range.append(test_cases[i][0])
#     plt.plot(x_range, y_range, label = 'Actual cases')
#     plt.plot(x_range, predictions, label = 'Predicted cases')
#     plt.xlabel("Day")
#     plt.ylabel("New Cases")
#     plt.title("Input Feature Used: Reproduction Rate, New Cases and New Deaths")
#     plt.legend()   
# =============================================================================













