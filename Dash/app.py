# -*- coding: utf-8 -*-
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output


app = dash.Dash()

df = pd.read_csv("stock_data.csv")
df2 = pd.read_csv("dataset_Facebook.csv",";")

if 'DYNO' in os.environ:
    app_name = os.environ['DASH_APP_NAME']
else:
    app_name = 'dash-timeseriesplot'


app.layout = html.Div([html.H1("Facebook Stock Prices Analysis", style={'textAlign': 'center'}),
    dcc.Dropdown(id='my-dropdown',options=[{'label': 'Tesla', 'value': 'TSLA'},{'label': 'Apple', 'value': 'AAPL'},{'label': 'Facebook', 'value': 'FB'},{'label': 'Microsoft', 'value': 'MSFT'}],
        multi=True,value=['FB'],style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "60%"}),
    dcc.Graph(id='my-graph'),
   html.H1("Facebook Metrics Distributions", style={"textAlign": "center"}),
   html.Div([html.Div([dcc.Dropdown(id='feature-selected1',
                                    options=[{'label': i.title(), 'value': i} for i in
                                             df2.columns.values[1:]],
                                    value="Type")], className="twelve columns",
                      style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "60%"}),
             ], className="row",
            style={"padding": 50, "width": "60%", "margin-left": "auto", "margin-right": "auto"}),
   dcc.Graph(id='my-graph2'),
], className="container")


@app.callback(Output('my-graph', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown_value:
        trace1.append(go.Scatter(x=df[df["Stock"] == stock]["Date"],y=df[df["Stock"] == stock]["Open"],mode='lines',
            opacity=0.7,name=f'Open {dropdown[stock]}',textposition='bottom center'))
        trace2.append(go.Scatter(x=df[df["Stock"] == stock]["Date"],y=df[df["Stock"] == stock]["Close"],mode='lines',
            opacity=0.6,name=f'Close {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
        'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
            height=600,title=f"Opening and Closing Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Price (USD)"})}
    return figure


@app.callback(
    dash.dependencies.Output('my-graph2', 'figure'),
    [dash.dependencies.Input('feature-selected1', 'value')])
def update_graph(selected_feature1):
    trace = go.Histogram(x=df2[selected_feature1],
                         marker=dict(color='rgb(0, 0, 100)'))

    return {
        'data': [trace],
        'layout': go.Layout(title=f'Metrics considered: {selected_feature1.title()}',
                            colorway=["#EF963B", "#EF533B"], hovermode="closest",
                            xaxis={'title': "Distribution", 'titlefont': {'color': 'black', 'size': 14},
                                   'tickfont': {'size': 14, 'color': 'black'}},
                            yaxis={'title': "Frequency", 'titlefont': {'color': 'black', 'size': 14, },
                                   'tickfont': {'color': 'black'}})}
if __name__ == '__main__':
    app.run_server(debug=True)
