# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 17:58:16 2021

@author: Adnan Labib
"""
# Importing required libraries
import requests
import json
from datetime import date, timedelta
import pandas as pd
import numpy as np

###################################################################
# Function used to get the past cases from the API
# API LINK: https://covidtracker.bsg.ox.ac.uk/about-api
# INPUT: country: the country for which past Covid-19 cases is needed
#        days: the number of days of Covid-19 cases data
# OUTPUT: a list containing the Covid-19 cases for the specified country
###################################################################


def get_cases(country, days, flag = True):
    
    # get today's date
    today = date.today()
    second_half = None
    
    # Some workarounds for countries that have issues with the API
    if country == 'Cote d\'Ivoire':
        country = 'Ivory Coast'
    # Multiple API calls needed for US, as one API call contains too much data
    if country == 'United States' and flag == True:
        second_half = get_cases("United States", 4, False)
        today = today-timedelta(5)
        days = 4
        
    # Send API call
    days = days + 3
    dateslist = [str(today - timedelta(days = day)) for day in range(days)]
    start = dateslist[-1]
    end = dateslist[1]
    url = f"https://api.covid19api.com/country/%s/status/confirmed?from=%sT00:00:00Z&to=%sT00:00:00Z" % (country,start,end)
    payload={}
    headers = {}
    
    # Get the response of the API call
    response = requests.request("GET", url, headers=headers, data=payload)
    
    # Parse JSON
    parsed_json = (json.loads(response.text))
    
    # Get the Number of Cases from the JSON response
    cases_list=[]
    if len(parsed_json)!=12:
        total = parsed_json[0]["Cases"]
        previous = None
        for item in range(1,len(parsed_json)):        
            if parsed_json[item]["Date"]==parsed_json[item-1]["Date"]:
                total += parsed_json[item]["Cases"]
            else:
                
                if previous!=None:
                    cases_list.append(total-previous)
                previous = total
                total = parsed_json[item]["Cases"]
        cases_list.append(total-previous)
    else:
        for cases in range(1,len(parsed_json)):        
            cases_list.append(parsed_json[cases]["Cases"]-parsed_json[cases-1]["Cases"])
        cases_list.pop(0)
        
    if second_half!=None:
        cases_list+=second_half
    
    
    return cases_list












