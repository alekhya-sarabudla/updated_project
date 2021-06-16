import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from plotly.graph_objs import *
import plotly.graph_objects as go
import pickle
from datetime import datetime
from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.holtwinters import ExponentialSmoothing

dataset = pd.read_csv('data3.csv')





dataset['x_time']=pd.to_datetime(dataset['x_time'],format="%Y-%m-%dT%H")

#dataset=dataset.set_index('x_time').resample('2T').mean().dropna().reset_index()

dataset['x_time'] = dataset['x_time'].dt.strftime("%Y-%m-%d %H:%M")

dataset['x_time']=pd.to_datetime(dataset['x_time'])



dataset=dataset.set_index('x_time').resample('2T').mean().dropna().reset_index()
datacopy= pd.DataFrame(columns=['x_time', 'apiTotalTime'])

#dataset=dataset.groupby(['x_time'],as_index=False).sum()
dataset.reset_index()


model_a = pickle.load(open('modell.pkl', 'rb'))


#dataset['apiTotalTime'] = dataset['apiTotalTime'].astype(np.uint8)



app = dash.Dash()

app.layout = html.Div(children=[
    html.H1(children='Visualisation Dashboard',style={'text-align':'center','color':'red'}),
     
    html.Button('Predict', id='submit-val', n_clicks=0),
   
    dcc.Graph(id='graph'),
    dcc.Graph(id='anomaly')
])

@app.callback(
    Output(component_id='graph', component_property='figure'),
    Output(component_id='anomaly', component_property='figure'),

    
    [dash.dependencies.Input('submit-val', 'n_clicks')],
   
)
def update_graph(n_clicks):
    

    
    dataset['anomaly']=dataset.apply(lambda x: 'Yes' if((x['apiTotalTime']>500) | (x['apiTotalTime']<80)) else 'No',axis=1)
    
    fig_a=px.scatter(dataset.reset_index(),x='x_time',y='apiTotalTime',color='anomaly')
    fig_a.update_layout(

     title={
        'text': "Anomaly Detection",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
        
        },
    
    xaxis_title=" ",
    yaxis_title=" ",
   
    font=dict(
        family="Times New Roman",
        size=18,
        color="blue"
         
    )
    
    
    )
    if(n_clicks>=1):
        
      
        
        model_a = pickle.load(open('modell_predict.pkl', 'rb'))
        forecast,err,ci = model_a
        n=len(forecast)
        df_forecast = pd.DataFrame({'forecast':forecast},index=pd.date_range(start='2020-09-20 10:36:16', periods=n, freq=' min'))
        fig=px.line(dataset,
                       x='x_time', y='apiTotalTime',
                       title='Forecasting')
        fig.add_trace(
        go.Scatter(
        x=df_forecast.index,
        y=df_forecast['forecast'],
        mode="lines",
        line=go.scatter.Line(color="red"),
        showlegend=False))

        fig.update_layout(

        
    
        xaxis_title=" ",
        yaxis_title=" "
        )
        
        
    else:
                

        fig=px.line(dataset,
                       x='x_time', y='apiTotalTime')
        fig.update_layout(

        
    
        xaxis_title=" ",
        yaxis_title=" "
        )
        
    return [fig,fig_a]


if __name__ == '__main__':
    app.run_server(debug=True)