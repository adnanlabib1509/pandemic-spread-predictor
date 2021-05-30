# -*- coding: utf-8 -*-
"""
Created on Sat May 22 17:26:45 2021

@author: Adnan Labib
"""
import glob
from tensorflow.keras.models import load_model, clone_model
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from api_getter import get_cases

##########################################################################
# This function is used to predict with all the models of the same group
# LSTM, GRU and Bi-LSTM groups
############################################################################

def choose_model(model_id, number_days, cases):
    if model_id==1:
        #print("LSTM Model is used")
        model_name = "vanilla_lstm"
    elif model_id==2:
        #print("Deep GRU Model is used")
        model_name = "deep_gru"
    else:
        #print("Bi-LSTM Model is used")
        model_name = "bi_lstm"
    file_list = []
    for filename in glob.glob('*.h5'):
        if model_name in filename:
            file_list.append(filename)

    value_list=[]
    new_value_list=[]
    mae_list=[] 
    #best_model = None       
    for i in range(len(file_list)):
        
        model = load_model(file_list[i])
        for times in range(int(number_days)):
            last_value = [cases[-1]]
            nfeatures = np.array(cases)
            nfeatures = nfeatures.reshape((1,10,1))
            predictions = model.predict(nfeatures)
            mae_list.append(mean_absolute_error(last_value, predictions))
            predictions = predictions.reshape(-1)
            value = predictions.tolist()[0]
            if value<0:
                value = 0
            value_list.append(value)
            cases.pop(0) 
            cases.append(value)
        #print(value_list)
        if len(file_list)==1:
            #best_model = file_list[i]
            return value_list
        else:
            #print("REACHED HERE")
            if i==0:
                current_mae = sum(mae_list)/len(mae_list)
                new_value_list = value_list.copy()
                #best_model = file_list[i]

            else:
                new_mae = sum(mae_list)/len(mae_list)
                if new_mae<current_mae:
                    new_value_list = value_list.copy()
                    #best_model = file_list[i]
                    #print(value_list)
            value_list.clear()
            mae_list.clear()    
        
    #print(new_value_list)  
    return new_value_list
                    
          
            
            
            
            
            