import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from statsmodels.tsa.arima_model import ARIMAResults

app = Flask(__name__)



model = pickle.load(open('modell.pkl', 'rb'))

model_hwes = pickle.load(open('model_hwes.pkl', 'rb'))
model_tbats = pickle.load(open('model_tbats.pkl', 'rb'))


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
    #results = loaded.fit()
    #output=loaded.summary()
   
   # prediction = model.forecast(steps=int_features)
    # output = round(prediction[0], 2)
    if(algo_value=="TBATS"):
        output=model_tbats.summary()
    elif(algo_value=="ARIMA"):
        output=loaded.summary()
    elif(algo_value=="HWES"):
        output=model_hwes.summary()

    return render_template('index.html', prediction_text='Summary of '+algo_value+' {}'.format(output))

#@app.route('/predict_api',methods=['POST'])
#def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)

#if __name__ == "__main__":
    #app.run(host='0.0.0.0',port=8080)