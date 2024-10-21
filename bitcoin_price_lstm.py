
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import plotly.graph_objects as go
plt.style.use ('fivethirtyeight')
import plotly.express as px

 
#!pip install pandas_datareader

 
#Reading data set and setting the index as Date
df = pd.read_csv('BTC-USD.csv')
df = df.set_index('Date')

 
#Droping null values
df=df.dropna()

 
#Showing the graph High, Low, Open, Close in Our dataset. x=Date y=INR
fig = go.Figure()
fig.add_trace(go.Scatter(x = df.index, y = df.High,mode='lines',name='High',marker_color = '#2CA02C',visible = "legendonly"))
fig.add_trace(go.Scatter(x = df.index, y = df.Low,mode='lines',name='Low',marker_color = '#D62728',visible = "legendonly"))
fig.add_trace(go.Scatter(x = df.index, y = df.Open,mode='lines',name='Open',marker_color = '#FF7F0E',visible = "legendonly"))
fig.add_trace(go.Scatter(x = df.index, y = df.Close,mode='lines',name='Close',marker_color = '#1F77B4'))

fig.update_layout(title='Closing price history',titlefont_size = 28,
                  xaxis = dict(title='Date',titlefont_size=16,tickfont_size=14),height = 800,
                  yaxis=dict(title='Price in INR (₹)',titlefont_size=16,tickfont_size=14),
                  legend=dict(y=0,x=1.0,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'))
fig.show()

  [markdown]
# # Preprocessing

 
data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * .8)

 
training_data_len

 
#MinMaxScaler Formula
#X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
#X_scaled = X_std * (max - min) + min

 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data

 
train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

 
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

  [markdown]
# # LSTM

 
model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error') #dog=input output=cat
model.fit(x_train, y_train, batch_size = 1, epochs = 1)

 
test_data = scaled_data[training_data_len - 60: , :]
x_test = []
y_test = dataset[training_data_len:, :]

for i in range (60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rsme = np.sqrt(np.mean(predictions - y_test) ** 2)
rsme

 
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

 
fig = go.Figure()
fig.add_trace(go.Scatter(x = train.index, y = train.Close,mode='lines',name='Close',marker_color = '#1F77B4'))
fig.add_trace(go.Scatter(x = valid.index, y = valid.Close,mode='lines',name='Val',marker_color = '#FF7F0E'))
fig.add_trace(go.Scatter(x = valid.index, y = valid.Predictions,mode='lines',name='Predictions',marker_color = '#2CA02C'))

fig.update_layout(title='Model',titlefont_size = 28,hovermode = 'x',
                  xaxis = dict(title='Date',titlefont_size=16,tickfont_size=14),height = 800,
                  yaxis=dict(title='Close price in INR (₹)',titlefont_size=16,tickfont_size=14),
                  legend=dict(y=0,x=1.0,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'))
fig.show()


