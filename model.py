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
    
    from statsmodels.tsa.holtwinters import ExponentialSmoothing




    
    dataset = pd.read_csv('data3.csv')
    dataset['x_time']=pd.to_datetime(dataset['x_time'],format="%Y-%m-%dT%H")
    dataset['x_time'] = dataset['x_time'].dt.strftime("%Y-%m-%d %H:%M")
    dataset['x_time']=pd.to_datetime(dataset['x_time'])
    





    datacopy= pd.DataFrame(columns=['x_time', 'apiTotalTime'])
    dataset=dataset.set_index('x_time').resample('2T').mean().dropna().reset_index()
    #dataset=dataset.groupby(['x_time'],as_index=False).sum()
    dataset.reset_index()
    print(dataset.head(100))

    start_date=dataset['x_time'].head(5)
    




#wtrain our modeldata.

    from statsmodels.tsa.arima_model import ARIMA
    model = ARIMA(dataset['apiTotalTime'], order=(1, 1,1))
    model_fit = model.fit()
    pickle.dump(model_fit, open('modell.pkl','wb'))

    model_a = pickle.load(open('modell_predict.pkl', 'rb'))
    forecast1,err,ci = model_a
    
    print(forecast1)
    print("original")
  

    
     





#------------hwes-----------

    model_hwes = ExponentialSmoothing(dataset['apiTotalTime'])
# fit the model
    model_hwes_fit=model_hwes.fit()
    forecast = model_hwes_fit.forecast(1000)
    #print(forecast)

    print(model_hwes_fit.predict(1000))


# Saving model to disk
    pickle.dump(model_hwes_fit, open('model_hwes.pkl','wb'))

#------------------tbats-------#

  


