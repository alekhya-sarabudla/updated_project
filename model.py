# Importing the libraries
if __name__ ==  '__main__':
    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.simplefilter('ignore', ConvergenceWarning)
    import numpy as np

    import matplotlib.pyplot as plt
    import pandas as pd
    import pickle
    from datetime import datetime
    import plotly.express as px
#from prophet import Prophet
    from statsmodels.tsa.holtwinters import ExponentialSmoothing




    dataset = pd.read_csv('data3.csv')




    dataset['x_time']=pd.to_datetime(dataset['x_time'],format="%Y-%m-%dT%H")


    dataset['x_time'] = dataset['x_time'].dt.strftime("%Y-%m-%d %H:%M")

    dataset['x_time']=pd.to_datetime(dataset['x_time'])





    datacopy= pd.DataFrame(columns=['x_time', 'apiTotalTime'])

    dataset=dataset.groupby(['x_time'],as_index=False).sum()
    dataset.reset_index()






#wtrain our modeldata.

    from statsmodels.tsa.arima_model import ARIMA
    model = ARIMA(dataset['apiTotalTime'], order=(1, 1,1))
    model_fit = model.fit()



# multi-step out-of-sample forecast
    forecast = model_fit.forecast(steps=7)



#Fitting model with trainig data

# Saving model to disk
    pickle.dump(model_fit, open('modell.pkl','wb'))

# Loading model to compare the results
#model = pickle.load(open('modell.pkl','rb'))
#print(model.predict([[2, 9, 6]]))

#-------PROPHET-----------------------------
#model_prophet = Prophet()
# fit the model
#model_prophet_fit=model_prophet.fit(dataset)


# Saving model to disk
#pickle.dump(model_prophet_fit, open('model_prophet.pkl','wb'))

# Loading model to compare the results
#model = pickle.load(open('model_prophet.pkl','rb'))
#print(model.predict([[2, 9, 6]]))
#--------------------------------------#--------------------


#------------hwes-----------

    model_hwes = ExponentialSmoothing(dataset['apiTotalTime'])
# fit the model
    model_hwes_fit=model_hwes.fit()


# Saving model to disk
    pickle.dump(model_hwes_fit, open('model_hwes.pkl','wb'))

#------------------tbats-------#

  


