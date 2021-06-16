import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from statsmodels.tsa.arima_model import ARIMAResults

app = Flask(__name__)



model = pickle.load(open('modell.pkl', 'rb'))

model_hwes = pickle.load(open('model_hwes.pkl', 'rb'))



loaded = ARIMAResults.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features =request.form['predictionduration']
    algo_value= request.form['algorithm']
    forecast=model.forecast(steps=int(int_features)*60)
    pickle.dump(forecast, open('modell_predict.pkl','wb'))
    model_a = pickle.load(open('modell_predict.pkl', 'rb'))
    forecast,err,ci = model_a
    
    #results = loaded.fit()
    #output=loaded.summary()
   
   # prediction = model.forecast(steps=int_features)
    # output = round(prediction[0], 2)
 
    if(algo_value=="ARIMA"):
        output=loaded.summary()
        #output=forecast
        
    elif(algo_value=="HWES"):
        output=model_hwes.summary()
        

    return render_template('output.html', prediction_text='Summary of '+algo_value+' {}'.format(output))

 


if __name__ == "__main__":
    app.run(debug=True)

#if __name__ == "__main__":
    #app.run(host='0.0.0.0',port=8080)