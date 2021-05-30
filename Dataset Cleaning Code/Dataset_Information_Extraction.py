# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 18:29:25 2021

@author: Adnan Labib
"""
import math
import numpy as np
import pandas as pd
import sys
import csv
from sklearn.impute import KNNImputer

# =============================================================================
# df = pd.read_csv("Current_Dataset.csv")
# 
# unique_values = df.location.unique()
# # =============================================================================
# # for i in range(len(unique_values)):
# #     print(f"<option value=\"%s\">%s</option>" % (unique_values[i],unique_values[i]))
# # =============================================================================
# df = df[["location","reproduction_rate"]]
# #stringency_index = df.groupby("location")["stringency_index"].tail(1) 
# stringency_index = df.groupby(['location']).last()
# #df = pd.DataFrame(list(zip(*[unique_values, stringency_index])))
# 
# stringency_index.to_csv('Final_DNN.csv')
# =============================================================================
df = pd.read_csv("Current_Dataset.csv")

unique_values = df.location.unique()
stringency_index = df.groupby("location")["stringency_index"].mean() 
population = df.groupby("location")["population"].mean()
population_density = df.groupby("location")["population_density"].mean()
median_age = df.groupby("location")["median_age"].mean()
reproduction_rate = df.groupby("location")["reproduction_rate"].mean()
#aged_65_older = df.groupby("location")["aged_65_older"].mean()
#aged_70_older = df.groupby("location")["aged_70_older"].mean()
#gdp_per_capita= df.groupby("location")["gdp_per_capita"].mean()
#cardiovasc_death_rate = df.groupby("location")["cardiovasc_death_rate"].mean()
#diabetes_prevalence = df.groupby("location")["diabetes_prevalence"].mean()
#handwashing_facilities = df.groupby("location")["handwashing_facilities"].mean()
#hospital_beds_per_thousand = df.groupby("location")["hospital_beds_per_thousand"].mean()
#life_expectancy = df.groupby("location")["life_expectancy"].mean()
#human_development_index = df.groupby("location")["human_development_index"].mean()

 
df = pd.DataFrame(list(zip(*[unique_values, population, population_density, 
                             median_age, stringency_index, reproduction_rate])))
df.to_csv('Final_DNN.csv', index=False)
   
# =============================================================================
# data = pd.read_csv("Final_DNN.csv")
# 
# imputer=KNNImputer(n_neighbors=3)
# imputed_data=pd.DataFrame(data=imputer.fit_transform(data),columns=[
#     "population", "population_density", "median_age", "stringency_index", 
#     "reproduction_rate", "latitude", "longitude", "best_model"])
# 
# imputed_data.to_csv('Final_DNN_New.csv', index=False)
# =============================================================================
