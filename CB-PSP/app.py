# Import Required Libraries
from flask import Flask,render_template,request,send_file,send_from_directory,jsonify, redirect
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from api_getter import get_cases
from datetime import date, timedelta
from stringency_index_api import str_index 
import pandas as pd
from model_training import training
from choose_model import choose_model
import matplotlib.pyplot as plt
import mpld3

# Initialize Flask
app = Flask(__name__,static_folder='static',template_folder='templates')

# Load the Population Density Dataset
p_density = pd.read_csv("population_density.csv")

# Load the three individual deep learning models
model_lstm = load_model('model_vanilla_lstm.h5', compile = False)
model_bi_lstm = load_model('model_bi_lstm.h5', compile = False)
model_deep_gru = load_model('model_deep_gru.h5', compile = False)

# Load the Bi-LSTM model that will generate the reproduction rate
model_reproduction_rate = load_model('model_reproduction_rate.h5', compile = False)

# Load the Model Selector
model_selector = pickle.load(open("rf_model.pkl", 'rb'))

test_list = []
train_list = []

##################################################################
# MAIN PAGE
##################################################################
@app.route('/',methods=['POST','GET'])
def main_page():
    if request.method=='GET':
        return render_template('main_page.html')
    if request.form['btn']=="Predict":
        return redirect("/predict")
    elif request.form['btn']=="Train":
        return redirect("/train")
    else:
        return redirect("/newdata")

##################################################################
# PREDICTION WITH USER'S OWN PANDEMIC CASES DATA
##################################################################
@app.route('/newdata',methods=['POST',"GET"])
def newdata():
    if request.method=='GET':
        return render_template('newdata.html')
    if request.form['btn']=="Back":
        return render_template('newdata.html')
    if request.method=='POST':

        country=[str(x) for x in request.form.values()]
        # Prepare the cases list [features = cases]
        features = country[:10]
        for number in range(len(features)):
            features[number]=int(features[number])
            
        # Prepare the three parameters for the model selector
        # Strigency Index, Population Density and Reproduction Rate
        stringency_index = country[10]
        pop_den = country[11]
        rep_rate = country[12]
        
        # Find out which deep learning model to use from the model selector
        values = np.array([float(stringency_index),float(pop_den),
                           float(rep_rate)]).reshape(-1,3)
        result = model_selector.predict(values)
        
        # Choose that deep learning model
        if result[0]==1:
            print("LSTM Model is used")
            model = model_lstm
        elif result[0]==2:
            print("Deep GRU Model is used")
            model = model_deep_gru
        else:
            print("Bi-LSTM Model is used")
            model = model_bi_lstm

        # Predict the future trends of the pandemic virus
        output_string = render_template('predict_result.html')
        value_list=[]
        day_list=[]
        for times in range(int(country[13])):
            nfeatures = np.array(features)
            nfeatures = nfeatures.reshape((1,10,1))
            predictions = model.predict(nfeatures)
            predictions = predictions.reshape(-1)
            value = predictions.tolist()[0]
        
            if value<0:
                value = 0
            value_list.append(value)
            day_list.append(times+1)
            features.pop(0) 
            features.append(value)
        
        # Plot the figure to be displayed on the webpage
        fig = plt.figure()
        plt.plot(day_list, value_list, label = 'Predicted cases', color='red')
        plt.xlabel("Day")
        plt.ylabel("Daily New Cases")
        plt.title("Daily Pandemic Cases for "+str(country[0]))
        plt.legend()
        plt.bar(day_list,value_list)
        
        # Convert the figure to be displayed on the webpage
        html = mpld3.fig_to_html(fig)             
        return output_string+html

##################################################################
# PREDICTION AFTER INCREMENTAL TRAINING WITH NEW DATA
##################################################################
@app.route('/train',methods=['POST', 'GET'])
def train():
    if request.method=='GET':
        return render_template('train.html')
    if request.form['btn']=="Back":
        return render_template('train.html')
    if request.method=='POST':
        country=[str(x) for x in request.form.values()]
        features = get_cases(country[0],10)
        if len(features)!=10:
            features.pop(0)
        # stringency index
        stringency_index = str_index(country[0])
        
        # population density
        for i in range(len(p_density)):
            if p_density["location"][i]==country[0]:
                pop_den = p_density["population_density"][i]
                print(p_density["location"][i])
                
        # reproduction rate
        rfeatures = np.array(features)
        rfeatures = rfeatures.reshape((1,10,1))
        rep_rate = model_reproduction_rate.predict(rfeatures)
        rep_rate = rep_rate.reshape(-1)
        rep_rate = rep_rate.tolist()[0]
        
        # Find out which deep learning model to use from the model selector
        values = np.array([float(stringency_index),float(pop_den),
                           float(rep_rate)]).reshape(-1,3)
        result = model_selector.predict(values)
        
        # Select the chosen deep learning model
        if result[0]==1:
            print("LSTM Model is used")
            model = model_lstm
            model_id = 1
        elif result[0]==2:
            print("Deep GRU Model is used")
            model = model_deep_gru
            model_id = 2
        else:
            print("Bi-LSTM Model is used")
            model = model_bi_lstm
            model_id = 3
        
        # Incremental Training on Existing Model
        if request.form['btn']=="Train on Existing":
            model = training(country[0], model_id, 1)
        # Generating New Model
        else:
            model = training(country[0], model_id, 2, int(country[2]))
        
        # Predict the future trends of the pandemic virus
        output_string = render_template('predict_result.html')
        value_list=[]
        day_list=[]
        for times in range(int(country[1])):
            nfeatures = np.array(features)
            nfeatures = nfeatures.reshape((1,10,1))
            predictions = model.predict(nfeatures)
            predictions = predictions.reshape(-1)
            value = predictions.tolist()[0]
        
            if value<0:
                value = 0
            value_list.append(value)
            day_list.append(times+1)
            features.pop(0) 
            features.append(value)
        
        # Plot the figure to be displayed on the webpage
        fig = plt.figure()
        plt.plot(day_list, value_list, label = 'Predicted cases', color='red')
        plt.xlabel("Day")
        plt.ylabel("Daily New Cases")
        plt.title("Daily Pandemic Cases for "+str(country[0]))
        plt.legend()
        plt.bar(day_list,value_list)
        
        # Convert the figure to be displayed on the webpage
        html = mpld3.fig_to_html(fig)
        return output_string+html

#################################################################
# DIRECT PREDICTION
#################################################################
@app.route('/predict',methods=['POST', 'GET'])
def predict():
    if request.method=='GET':
        return render_template('predict.html')
    if request.form['btn']=="Back":
        return render_template('predict.html')   
    if request.method=='POST':
        country=[str(x) for x in request.form.values()]
        features = get_cases(country[0],10)
        if len(features)!=10:
            features.pop(0)
        
        # stringency index
        stringency_index = str_index(country[0])
        
        # population
        for i in range(len(p_density)):
            if p_density["location"][i]==country[0]:
                pop_den = p_density["population_density"][i]
                print(p_density["location"][i])
                
        # reproduction rate
        rfeatures = np.array(features)
        rfeatures = rfeatures.reshape((1,10,1))
        rep_rate = model_reproduction_rate.predict(rfeatures)
        rep_rate = rep_rate.reshape(-1)
        rep_rate = rep_rate.tolist()[0]
        
        # Find out which deep learning model to use from the model selector
        values = np.array([float(stringency_index),float(pop_den),
                           float(rep_rate)]).reshape(-1,3)
        result = model_selector.predict(values)
        
        # Select the chosen deep learning model
        if result[0]==1:
            print("LSTM Model is used")
            model_id = 1
        elif result[0]==2:
            print("Deep GRU Model is used")
            model_id = 2
        else:
            print("Bi-LSTM Model is used")
            model_id = 3
        
        # Predict the future trends of the pandemic virus
        output_string = render_template('predict_result.html')
        value_list = choose_model(model_id,country[1],features)
        day_list = []     
        for i in range(len(value_list)):
            day_list.append(i+1)
        
        # Plot the figure to be displayed on the webpage
        fig = plt.figure()
        plt.plot(day_list, value_list, label = 'Predicted cases', color='red')
        plt.xlabel("Day")
        plt.ylabel("Daily New Cases")
        plt.title("Daily Pandemic Cases for "+str(country[0]))
        plt.legend()
        plt.bar(day_list,value_list)
        
        # Convert the figure to be displayed on the webpage
        html = mpld3.fig_to_html(fig)
        return output_string+html

if __name__=='__main__':
  app.run(host='localhost',port=5000)