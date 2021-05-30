# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 23:48:20 2021

@author: Adnan Labib
"""
import math
import numpy as np
import pandas as pd

df = pd.read_csv("Covid_Dataset_31_3.csv")

omit_ls = ['Angola', 'Anguilla', 'Belize', 'Bermuda', 'Botswana', 'Burundi',
           'Cape Verde', 'Cayman Islands', 'Comoros', 'Dominica', 'Eritrea',
           'Faeroe Islands', 'Falkland Islands', 'Greenland', 'Gibraltar', 
           'Grenada', 'Guernsey', 'Guinea-Bissau', 'Haiti', 'Hong Kong', 
           'Isle of Man', 'Jersey', 'Laos', 'Lesotho', 'Macao', 'Madagascar',
           'Malawi', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 
           'Micronesia (country)', 'Montserrat','Mozambique', 'Myanmar', 'Niger', 
           'Northern Cyprus', 'Papua New Guinea', 'Saint Helena', 
           'Saint Kitts and Nevis', 'Saint Lucia',
           'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 
           'Sao Tome and Principe', 'Sierra Leone', 'Solomon Islands', 
           'South Sudan', 'Syria', 'Tajikistan', 'Timor', 'Togo', 
           'Trinidad and Tobago', 'Turks and Caicos Islands', 'Uganda',
           'Vanuatu', 'Yemen', 'Zimbabwe', 'Africa', 'Antigua and Barbuda',
           'Armenia', 'Asia', 'Equatorial Guinea', 'Europe', 'European Union',
           'Leichtenstein', 'Maldives', 'Montenegro', 'North America',
           'North Macedonia', 'Oceania', 'South America', 'Vatican', 'World',
           'Liechtenstein', 'International']

for i in range(len(omit_ls)):
    print(i)
    index_names = df[ df['location'] == omit_ls[i]].index
    df = df.drop(index_names)
    
df.to_csv('Current_Dataset.csv', index = False)