# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

external_scripts = ['/assets/style.css']

app = dash.Dash(__name__,
                external_scripts=external_scripts)
server = app.server

df = pd.read_csv("stock_data.csv")
df2 = pd.read_csv("dataset_Facebook.csv",";")

df_ml = df2.copy()

lb_make = LabelEncoder()
df_ml["Type"] = lb_make.fit_transform(df_ml["Type"])
df_ml = df_ml.fillna(0)

X = df_ml.drop(['like'], axis = 1).values
Y = df_ml['like'].values

X = StandardScaler().fit_transform(X)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.30, random_state = 101)

randomforest = RandomForestRegressor(n_estimators=500,min_samples_split=10)
randomforest.fit(X_Train,Y_Train)

p_train = randomforest.predict(X_Train)
p_test = randomforest.predict(X_Test)

train_acc = r2_score(Y_Train, p_train)
test_acc = r2_score(Y_Test, p_test)

app.layout = html.Div([html.H1("Facebook Data Analysis", style={"textAlign": "center"}), dcc.Markdown('''
Welcome to my Plotly (Dash) Data Science interactive dashboard. In order to create this dashboard have been used two different datasets. The first one is the [Huge Stock Market Dataset by Boris Marjanovic](https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)
and the second one is the [Facebook metrics Data Set by Moro, S., Rita, P., & Vala, B](https://archive.ics.uci.edu/ml/datasets/Facebook+metrics). This dashboard is divided in 3 main tabs. In the first one you can choose whith which other companies to compare Facebook Stock Prices to anaylise main trends.
Using the second tab, you can analyse the distributions each of the Facebook Metrics Data Set features. Particular interest is on how paying to advertise posts can boost posts visibility. Finally, in the third tab a Machine Learning analysis of the considered datasets is proposed. 
All the data displayed in this dashboard is fetched, processed and updated using Python (eg. ML models are trained in real time!).
''')  ,
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Stock Prices', children=[
html.Div([html.H1("Dataset Introduction", style={'textAlign': 'center'}),
dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in df.columns],
    data=df.iloc[0:5,:].to_dict("rows"),
),
    html.H1("Facebook Stocks High vs Lows", style={'textAlign': 'center', 'padding-top': 5}),
    dcc.Dropdown(id='my-dropdown',options=[{'label': 'Tesla', 'value': 'TSLA'},{'label': 'Apple', 'value': 'AAPL'},{'label': 'Facebook', 'value': 'FB'},{'label': 'Microsoft', 'value': 'MSFT'}],
        multi=True,value=['FB'],style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "80%"}),
    dcc.Graph(id='highlow'),  dash_table.DataTable(
    id='table2',
    columns=[{"name": i, "id": i} for i in df.describe().reset_index().columns],
    data= df.describe().reset_index().to_dict("rows"),
),
    html.H1("Facebook Market Volume", style={'textAlign': 'center', 'padding-top': 5}),
    dcc.Dropdown(id='my-dropdown2',options=[{'label': 'Tesla', 'value': 'TSLA'},{'label': 'Apple', 'value': 'AAPL'},{'label': 'Facebook', 'value': 'FB'},{'label': 'Microsoft', 'value': 'MSFT'}],
        multi=True,value=['FB'],style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "80%"}),
    dcc.Graph(id='volume'),
    html.H1("Scatter Analysis", style={'textAlign': 'center', 'padding-top': -10}),
    dcc.Dropdown(id='my-dropdown3',
                 options=[{'label': 'Tesla', 'value': 'TSLA'}, {'label': 'Apple', 'value': 'AAPL'},
                          {'label': 'Facebook', 'value': 'FB'}, {'label': 'Microsoft', 'value': 'MSFT'}],
                 value= 'FB',
                 style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "45%"}),
    dcc.Dropdown(id='my-dropdown4',
                 options=[{'label': 'Tesla', 'value': 'TSLA'}, {'label': 'Apple', 'value': 'AAPL'},
                          {'label': 'Facebook', 'value': 'FB'}, {'label': 'Microsoft', 'value': 'MSFT'}],
                 value= 'AAPL',
                 style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "45%"}),
  dcc.RadioItems(id="radiob", value= "High", labelStyle={'display': 'inline-block', 'padding': 10},
                 options=[{'label': "High", 'value': "High"}, {'label': "Low", 'value': "Low"} , {'label': "Volume", 'value': "Volume"}],
 style={'textAlign': "center", }),
    dcc.Graph(id='scatter')
], className="container"),
]),
dcc.Tab(label='Performance Metrics', children=[
html.Div([html.H1("Facebook Metrics Distributions", style={"textAlign": "center"}),
            html.Div([html.Div([dcc.Dropdown(id='feature-selected1',
                                             options=[{'label': i.title(), 'value': i} for i in
                                                      df2.columns.values[1:]],
                                             value="Type")],
                               style={"display": "block", "margin-left": "auto", "margin-right": "auto",
                                      "width": "80%"}),
                      ],),
            dcc.Graph(id='my-graph2'),
dash_table.DataTable(
    id='table3',
    columns=[{"name": i, "id": i} for i in df.describe().reset_index().columns],
    data= df.describe().reset_index().to_dict("rows"),
),
            html.H1("Paid vs Free Posts by Category", style={'textAlign': "center", 'padding-top': 5}),
     html.Div([
         dcc.RadioItems(id="select-survival", value=str(1), labelStyle={'display': 'inline-block', 'padding': 10},
                        options=[{'label': "Paid", 'value': str(1)}, {'label': "Free", 'value': str(0)}], )],
         style={'textAlign': "center", }),
     html.Div([html.Div([dcc.Graph(id="hist-graph", clear_on_unhover=True, )]), ]),
        ], className="container"),
]),
dcc.Tab(label='Machine Learning', children=[
html.Div([html.H1("Machine Learning", style={"textAlign": "center"}), html.H2("ARIMA Time Series Prediction", style={"textAlign": "left"}),
    dcc.Dropdown(id='my-dropdowntest',options=[{'label': 'Tesla', 'value': 'TSLA'},{'label': 'Apple', 'value': 'AAPL'},{'label': 'Facebook', 'value': 'FB'},{'label': 'Microsoft', 'value': 'MSFT'}],
                style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "50%"}),
          dcc.RadioItems(id="radiopred", value="High", labelStyle={'display': 'inline-block', 'padding': 10},
                         options=[{'label': "High", 'value': "High"}, {'label': "Low", 'value': "Low"},
                                  {'label': "Volume", 'value': "Volume"}], style={'textAlign': "center", }),
    dcc.Graph(id='traintest'), dcc.Graph(id='preds'),
html.H2("Performance Metrics Regression Prediction", style={"textAlign": "left"}), html.P("In this example I used the Facebook Performance Metrics dataset to predict the number of likes I post can get. Training a Random Forest Regressor with 500 estimetors right now online lead an accuracy (%) in the Training set equal to: "),
    str(train_acc), html.P("In the Test set, was instead registred an accuracy (%) of:"), str(test_acc),
    html.P("In order to achieve these results, all the not a numbers (NaNs) have been eliminated, categorical data has been encoded and the data has been normalized. The R2 score has been used as metric for this exercise and a Train/Test split ratio of 70:30% was used.")],)
], className="container")
])
])


@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(go.Scatter(x=df[df["Stock"] == stock]["Date"],y=df[df["Stock"] == stock]["High"],mode='lines',
            opacity=0.7,name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(go.Scatter(x=df[df["Stock"] == stock]["Date"],y=df[df["Stock"] == stock]["Low"],mode='lines',
            opacity=0.6,name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
        'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
            height=600,title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Price (USD)"},     paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)')}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(go.Scatter(x=df[df["Stock"] == stock]["Date"],y=df[df["Stock"] == stock]["Volume"],mode='lines',
            opacity=0.7,name=f'Volume {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
        'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
            height=600,title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Transactions Volume"} ,   paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)')}
    return figure


@app.callback(Output('scatter', 'figure'),
              [Input('my-dropdown3', 'value'), Input('my-dropdown4', 'value'), Input("radiob", "value"),])
def update_graph(stock, stock2, radioval):
    dropdown = {"TSLA": "Tesla", "AAPL": "Apple", "FB": "Facebook", "MSFT": "Microsoft", }
    radio = {"High": "High Prices", "Low": "Low Prices", "Volume": "Market Volume", }
    trace1 = []
    if (stock == None) or (stock2 == None):
        trace1.append(
            go.Scatter(x= [0], y= [0],
                       mode='markers', opacity=0.7, textposition='bottom center'))
        traces = [trace1]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
                  'layout': go.Layout(colorway=['#FF7400', '#FFF400', '#FF0056'],
                                      height=600, title=f"{radio[radioval]}",
                                      paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)')}
    else:
        trace1.append(go.Scatter(x=df[df["Stock"] == stock][radioval][-1000:], y=df[df["Stock"] == stock2][radioval][-1000:],
                       mode='markers', opacity=0.7, textposition='bottom center'))
        traces = [trace1]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=['#FF7400', '#FFF400', '#FF0056'],
                height=600,title=f"{radio[radioval]} of {dropdown[stock]} vs {dropdown[stock2]} Over Time (1000 iterations)",
                xaxis={"title": stock,}, yaxis={"title": stock2},     paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')}
    return figure


@app.callback(
    dash.dependencies.Output('my-graph2', 'figure'),
    [dash.dependencies.Input('feature-selected1', 'value')])
def update_graph(selected_feature1):
    if selected_feature1 == None:
        selected_feature1 = 'Type'
        trace = go.Histogram(x= df2.Type,
                             marker=dict(color='rgb(0, 0, 100)'))
    else:
        trace = go.Histogram(x=df2[selected_feature1],
                         marker=dict(color='rgb(0, 0, 100)'))

    return {
        'data': [trace],
        'layout': go.Layout(title=f'Metric: {selected_feature1.title()}',
                            colorway=["#EF963B", "#EF533B"], hovermode="closest",
                            xaxis={'title': "Distribution", 'titlefont': {'color': 'black', 'size': 14},
                                   'tickfont': {'size': 14, 'color': 'black'}},
                            yaxis={'title': "Frequency", 'titlefont': {'color': 'black', 'size': 14, },
                                   'tickfont': {'color': 'black'}},     paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)')}


@app.callback(
    dash.dependencies.Output("hist-graph", "figure"),
    [dash.dependencies.Input("select-survival", "value"),])
def update_graph(selected):
    dff = df2[df2["Paid"] == int(selected)]
    trace = go.Histogram(x=dff["Type"], marker=dict(color='rgb(0, 0, 100)'))
    layout = go.Layout(xaxis={"title": "Post distribution categories", "showgrid": False},
                       yaxis={"title": "Frequency", "showgrid": False},    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)' )
    figure2 = {"data": [trace], "layout": layout}

    return figure2


@app.callback(Output('traintest', 'figure'),
              [Input('my-dropdowntest', 'value'), Input("radiopred", "value"),])
def update_graph(stock , radioval):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    radio = {"High": "High Prices", "Low": "Low Prices", "Volume": "Market Volume", }
    trace1 = []
    trace2 = []
    train_data = df[df['Stock'] == stock][-1000:][0:int(1000 * 0.8)]
    test_data = df[df['Stock'] == stock][-1000:][int(1000 * 0.8):]
    if (stock == None):
        trace1.append(
            go.Scatter(x= [0], y= [0],
                       mode='markers', opacity=0.7, textposition='bottom center'))
        traces = [trace1]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
                  'layout': go.Layout(colorway=['#FF7400', '#FFF400', '#FF0056'],
                                      height=600, title=f"{radio[radioval]}",
                                      paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)')}
    else:
        trace1.append(go.Scatter(x=train_data['Date'],y=train_data[radioval], mode='lines',
            opacity=0.7,name=f'Training Set',textposition='bottom center'))
        trace2.append(go.Scatter(x=test_data['Date'],y=test_data[radioval],mode='lines',
            opacity=0.6,name=f'Test Set',textposition='bottom center'))
        traces = [trace1, trace2]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                height=600,title=f"{radio[radioval]} Train-Test Sets for {dropdown[stock]}",
                xaxis={"title":"Date",
                       'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'step': 'all'}])},
                       'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Price (USD)"},     paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')}
    return figure


@app.callback(Output('preds', 'figure'),
              [Input('my-dropdowntest', 'value'), Input("radiopred", "value"),])
def update_graph(stock, radioval):
    dropdown = {"TSLA": "Tesla", "AAPL": "Apple", "FB": "Facebook", "MSFT": "Microsoft", }
    radio = {"High": "High Prices", "Low": "Low Prices", "Volume": "Market Volume", }
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    if (stock == None):
        trace1.append(
            go.Scatter(x= [0], y= [0],
                       mode='markers', opacity=0.7, textposition='bottom center'))
        traces = [trace1]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
                  'layout': go.Layout(colorway=['#FF7400', '#FFF400', '#FF0056'],
                                      height=600, title=f"{radio[radioval]}",
                                      paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)')}
    else:
        test_data = df[df['Stock'] == stock][-1000:][int(1000 * 0.8):]
        train_data = df[df['Stock'] == stock][-1000:][0:int(1000 * 0.8)]
        train_ar = train_data[radioval].values
        test_ar = test_data[radioval].values
        history = [x for x in train_ar]
        predictions = list()
        for t in range(len(test_ar)):
            model = ARIMA(history, order=(3, 1, 0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test_ar[t]
            history.append(obs)
        error = mean_squared_error(test_ar, predictions)
        trace1.append(go.Scatter(x=test_data['Date'],y=test_data['High'],mode='lines',
            opacity=0.6,name=f'Actual Series',textposition='bottom center'))
        trace2.append(go.Scatter(x=test_data['Date'],y= np.concatenate(predictions).ravel(), mode='lines',
            opacity=0.7,name=f'Predicted Series (MSE: {error})',textposition='bottom center'))
        traces = [trace1, trace2]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                height=600,title=f"{radio[radioval]} ARIMA Predictions vs Actual for {dropdown[stock]}",
                xaxis={"title":"Date",
                       'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'step': 'all'}])},
                       'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Price (USD)"},     paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')}
    return figure


if __name__ == '__main__':
    app.run_server(debug=True)
