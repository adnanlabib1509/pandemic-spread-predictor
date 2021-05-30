# -*- coding: utf-8 -*-
"""
Created on Wed May  5 19:44:22 2021

@author: Adnan Labib
"""
import requests
import json
from datetime import date, timedelta
import sys
from iso3166 import countries
import pandas as pd
import numpy as np

def str_index(given_country):
    
    
    
    if given_country == 'Bolivia':
        country = 'BOL'
    elif given_country == 'Brunei':
        country = 'BRN'
    elif given_country == "Cote d'Ivoire":
        country = 'CIV'
    elif given_country == 'Iran':
        country = 'IRN'
    elif given_country == 'Moldova':
        country = 'MDA'
    elif given_country == 'Russia':
        country = 'RUS'
    elif given_country == 'South Korea':
        country = 'KOR'
    elif given_country == 'Tanzania':
        country = 'TZA'
    elif given_country == 'United Kingdom':
        country = 'GBR'
    elif given_country == 'United States':
        country = 'USA'
    elif given_country == 'Venezuela':
        country = 'VEN'
    elif given_country == 'Vietnam':
        country = 'VNM'
    else:
        country = str(countries.get(given_country).alpha3)
    today = str(date.today()-timedelta(2))
        
    url = f"https://covidtrackerapi.bsg.ox.ac.uk/api/v2/stringency/actions/{country}/{today}"
    
    payload={}
    headers = {}
    
    response = requests.request("GET", url, headers=headers, data=payload)
    
    parsed_json = (json.loads(response.text))
    #print(json.dumps(parsed_json, indent=4, sort_keys=True))
    return parsed_json["stringencyData"]["stringency"]




