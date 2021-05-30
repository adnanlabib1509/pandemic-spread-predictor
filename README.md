# Cloud Based Pandemic Spread Predictor (CB-PSP)

Cloud Based Pandemic Spread Predictor (CB-PSP) is a cloud framework comprising of deep learning models which are used to predict the spread of a pandemic level virus, by
analyzing the positive cases trends.

## Table of Contents

```bash
CB-PSP: contains the main app of the project
Dataset Cleaning Code: contains the code used to clean the dataset
Datasets: contains all the datasets used in this project
Existing Models: contains the existing models used for comparison in this project
Experimentation Results: contains the country-wise results obtained in this project
Graph Results: Contains the country-wise graph results
Model_Training_Testing: Contains the code used for training and testing the models
Reproduction Rate Models: contains the models considered for predicting the reproduction rate
Swine Flu Experiment: Contains all the code and models used for the swine flu generalization experiment
Used_Model_Code: Contains the model generation code
AWS_Cloud_Screenshot.docx: Contains some screenshots of AWS component creation using AWS Management Console
Setup_Locally.docx: file containing the steps to follow to set up the system locally, which is also explained below
```

## Installation

The file "CB-PSP" contains the code used for the system.

### Pre-requisites

```bash
python >= 3.5
Python Anaconda Distribution
```

### Local Setup
Step 1: Open Anaconda Prompt

Step 2: Cd into the CB-PSP folder

```bash
cd CB-PSP
```

Step 3: Install the supporting libraires

```bash
pip install flask
pip install pickle
pip install numpy
pip install tensorflow
pip install keras
pip install datetime
pip install pandas
pip install matplotlib
pip install mpld3
pip install requests
pip install json
pip install glob
pip install sklearn
pip install iso3166
```
Step 4: Run the app using:

```bash
app.py
```
This will run the application on http://localhost:5000/
Copy paste the link on the browser address location and the home page will pop up.





