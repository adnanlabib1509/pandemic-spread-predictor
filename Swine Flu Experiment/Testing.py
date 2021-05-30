# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 13:10:52 2021

@author: Adnan Labib
"""
# Importing Required Models
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
import pickle

model_lstm = load_model('model_vanilla_lstm.h5')
model_deep_gru = load_model('model_deep_gru.h5')
model_bi_lstm = load_model('model_bi_lstm.h5')

df = pd.read_csv("Swine_Flu_Dataset.csv")

# convert all NaN values to zero
df = df.fillna(0)

# =============================================================================
# world_df=df[(df.country=="United States")]
# world_df.to_csv('Final_DNN.csv')
# sys.exit()
# =============================================================================

#all_countries = ['Malaysia']
#print(all_countries[70])
#sys.exit()

# =============================================================================
# all_countries = df.country.unique()
# print(all_countries)
# sys.exit()
# =============================================================================
world_df=df[(df.country=="Mexico")]
total_cases = world_df["cases"].values.tolist()


# Population Density
pop_val = 57

new_cases = []
for i in range(1,len(total_cases)):
    if total_cases[i]-total_cases[i-1]!=0:
        new_cases.append(total_cases[i]-total_cases[i-1])
    else:
        if i!=(len(total_cases)-1):
            new_cases.append(int(abs((total_cases[i+1]-total_cases[i-1])/2)))
     

# =============================================================================
# # Getting the training data
# x_train=[]
# y_train = []
# 
# test_cases = np.array(new_cases).reshape(-1,1)
# for i in range(10):
#     for j in range(10):
#         x_train.append(test_cases[i+j])
#     y_train.append(test_cases[i+10,0])
# 
# # Convert to numpy array
# x_train, y_train = np.array(x_train), np.array(y_train)
# x_train = x_train.reshape((10,10,1))
# =============================================================================


# Getting the testing data
x_test=[]
y_test = []

test_cases = np.array(new_cases).reshape(-1,1)
for i in range(30):
    for j in range(10):
        x_test.append(test_cases[i+j])
    y_test.append(test_cases[i+10,0])

# Convert to numpy array
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = x_test.reshape((30,10,1))

# Stringency Index
str_model = load_model("model_stringency_index.h5")
str_pred = str_model.predict(x_test)
str_value = sum(str_pred)/len(str_pred)
str_val = str_value[0]
print(str_val)

# Reproduction Rate
repro_model = load_model("model_reproduction_rate.h5")
repro_pred = repro_model.predict(x_test)
repro_value = sum(repro_pred)/len(repro_pred)
rep_val = repro_value[0]
print(rep_val)



model_selector = pickle.load(open("rf_model.pkl", 'rb'))
model_id = model_selector.predict(np.array([str_val,pop_val,rep_val]).reshape(-1,3))
print(model_id)

if model_id[0]==1:
    model = model_lstm
elif model_id[0]==2:
    model = model_deep_gru
else:
    model = model_bi_lstm

#model = model_bi_lstm

# =============================================================================
# model.compile(optimizer = "Adam", loss = 'mean_squared_error') 
# model.fit(x_train, y_train, epochs = 250, batch_size = 10, verbose = 0)
# =============================================================================

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
mae = mean_absolute_error(y_test,predictions)
#rmse = math.sqrt(mse)
print(" MAE: ", mae)

#mae_list.append(mae)

#predictions = predictions.reshape(-1)
#print(predictions.tolist()[0])

#print(sum(mae_list)/len(mae_list))

    
# Get the graph showing the comparison between predicted and actual 
x_range=[]
y_range=[]
#test_date = df[["date"]].values.tolist()
        

for i in range(10,40):
    x_range.append(i-10)
    y_range.append(test_cases[i][0])

plt.plot(x_range, y_range, label = 'Actual cases')
plt.plot(x_range, predictions, label = 'Predicted cases')
#plt.ylim((0,1000))
plt.xlabel("Day")
plt.ylabel("Daily New Cases")
plt.title("Daily H1N1 Virus Cases for Mexico")
plt.legend()
plt.savefig('Mexico.png', dpi=300)   













