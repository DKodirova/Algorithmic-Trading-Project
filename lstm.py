import warnings
warnings.filterwarnings('ignore')
import math
import pandas as pd
import pandas_datareader as web
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM # type: ignore
from keras.layers import Dense # type: ignore
from keras.models import Sequential # type: ignore

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def strategy2(n):
    print("current n is:", n)
    data_frame = pd.read_csv('data/AAPL-3.csv')
    data_frame['Date'] = pd.to_datetime(data_frame['Date'])
    
    latest_date = data_frame['Date'].max()
    start_date = latest_date - pd.DateOffset(months=int(n * 12))
    data_frame = data_frame[data_frame['Date'] >= start_date]
    
    stock_close_data = data_frame.filter(['Close'])
    stock_close_dataset = stock_close_data.values
    trainingDataLength = math.ceil( len(stock_close_dataset) * 0.8 )
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaledData = scaler.fit_transform(stock_close_dataset)
    
    StockTrainData = scaledData[0:trainingDataLength , :]
    Xtrain = []
    Ytrain = []

    for i in range(60, len(StockTrainData)):
        Xtrain.append(StockTrainData[i-60:i, 0])
        Ytrain.append(StockTrainData[i, 0])

    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))
    
    model = Sequential()

    neurons = 50

    model.add(LSTM(neurons, return_sequences=True, input_shape= (Xtrain.shape[1], 1))) 

    model.add(LSTM(neurons, return_sequences= False)) 

    model.add(Dense(25)) 
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse') 
    history_data = model.fit(Xtrain, Ytrain, batch_size=50, epochs=200, verbose=2, validation_split=0.2)
    
    testingData = scaledData[trainingDataLength - 60: , :]

    Xtest = []
    Ytest = stock_close_dataset[trainingDataLength:, :]
    for i in range(60, len(testingData)):
        Xtest.append(testingData[i-60:i, 0])
        
    Xtest = np.array(Xtest)
    Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1 ))
    predictions = model.predict(Xtest)
    predictions = scaler.inverse_transform(predictions)

if __name__ == "__main__":
    strategy2(0.5)