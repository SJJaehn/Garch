import pandas as pd
import numpy as np
import os, random, time

from util import *

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, InputLayer
from keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore')

# binance_agg for the aggregated trade data (used in the paper)
# binance_candles for the candlestick data
# kraken_candles for the candlestick data from Kraken
data_source = 'kraken_candles'

lookback = 24
train_size = 0.5
val_size = 0.25

min_recall = 0.08
max_recall = 0.15

random_states = range(10)

def create_model(input_shape):
    model = Sequential()
    model.add(InputLayer(input_shape=(input_shape)))
    model.add(LSTM(8, return_sequences=False, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
    
    
start = time.time()

# Creating a dataframe to store the results
result = pd.DataFrame(columns = ['source', 'model', 'dataset', 'extended', 'val_pre', 'val_recall', 'val_return', 'val_sharp', 'test_pre', 'test_recall', 'test_return', 'test_sharp'])
# Dataframe for the significance test
df_t_test = pd.DataFrame(columns = ['source', 'model', 'dataset1', 'dataset2', "p-value"])

# Reading the data
df = pd.read_csv('./Data/' + data_source + '.csv') # Reading the data from the csv file
df['timestamp'] = pd.to_datetime(df['timestamp']) # Converting the timestamp to datetime
df.dropna(inplace=True) # Dropping all samples with NaN values
df.set_index('timestamp', inplace=True) # Setting the timestamp as the index

# Calculating the indicators
df = get_indicators(df)
df = get_cyclical_features(df)
df = get_candle_features(df)

# Calculating the performance within the next periode
df['perf'] = df['close'].pct_change().shift(-1)

# Calculating the label based on the performance
df['y'] = (df['perf'] > 0).astype(int) # When the performance is greater than 0, the label is 1 (buy), otherwise 0 (hold)

# Dropping the first 29 rows to avoid NaN values from the indicators
df = df[29:43465]

# Splitting the features into the three groups
ohlc = df[['open', 'high', 'low', 'close', 'volume', 'number_of_trades', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos']]
candle = df[['close', 'body', 'color', 'upper_shadow', 'lower_shadow', 'volume', 'number_of_trades', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos']]
extended_df = df[['sma_15', 'sma_20', 'sma_25', 'sma_30', 'rsi_15', 'rsi_20', 'rsi_25', 'rsi_30', 'williams_r_14', 'macd_12_26', 'mfi_14', 'so_14']]

# Creating the output directory
output_dir = f'./Results/LSTM-paper-keras/{data_source}'
os.makedirs(output_dir, exist_ok=True)

for random_state in random_states:

    os.environ['PYTHONHASHSEED']=str(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    
    print(f"Run {random_state + 1}/{len(random_states)}")
    
    # Temporary dictionary
    dic_performace = {}

    # Go through all combinations
    for base in [ohlc, candle]:
        for extended in [False, True]:
    
            # Creating the dataset
            data = base.copy()
            if extended:
                data = pd.concat([data, extended_df.copy()], axis=1)
            
            # Getting the performance for later calculations and y
            perf = df['perf'].copy()
            y = df['y'].copy()
    
            # Creating the sequences with the lookback period
            X, y, perf = create_sequence(data, y, perf, lookback)
    
            # Splitting the data into train, validation and test sets
            end_train = int(len(X) * train_size)
            end_val = int(len(X) * (train_size + val_size))
    
            X_train, y_train, perf_train = X[:end_train], y[:end_train], perf[:end_train]
            X_val, y_val, perf_val = X[end_train:end_val], y[end_train:end_val], perf[end_train:end_val]
            X_test, y_test, perf_test = X[end_val:], y[end_val:], perf[end_val:]
    
            # Using a MinMaxScaler to rescale the data
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
            X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
            # Creating the model
            model = create_model((lookback, X_train.shape[2]))
    
            # Using early stopping to avoid overfitting as mentioned in the paper
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)         
    
            # Training the model
            model.fit(X_train, y_train, epochs=500, batch_size=1024, validation_data=(X_val, y_val), callbacks=[early_stopping],verbose=0)
    
            # Predicting the validation and test set
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
    
            # Calculating the threshold
            threshold = find_threshold(y_val_pred, y_val, min_recall, max_recall)
    
            # Calculating the metrics
            val_sharp, val_pre, val_recall, val_return, fig_val = calculate_metrics(y_val, y_val_pred, perf_val, threshold)
            test_sharp, test_pre, test_recall, test_return, fig_test = calculate_metrics(y_test, y_test_pred, perf_test, threshold)

            # Adding the performance to the dictionary for the significance test
            temp = ((y_test_pred >= threshold).astype(int)).flatten() * perf_test
            dic_performace[f'{'ohlc' if base.equals(ohlc) else 'candle'} - {'extended' if extended else 'raw'}'] = temp

            result = result._append({'source': data_source, 'model': 'LSTM-paper-keras', 'dataset' : 'ohlc' if base.equals(ohlc) else 'candle', 'extended' : extended, 'val_pre' : val_pre, 'val_recall' : val_recall, 'val_return' : val_return, 'val_sharp' : val_sharp, 'test_pre' : test_pre, 'test_recall' : test_recall, 'test_return' : test_return, 'test_sharp' : test_sharp}, ignore_index=True)


            fig_val.savefig(f'{output_dir}/val_{random_state}_{'ohlc' if base.equals(ohlc) else 'candle'} - {'extended' if extended else 'raw'}.png')
            fig_test.savefig(f'{output_dir}/test_{random_state}_{'ohlc' if base.equals(ohlc) else 'candle'} - {'extended' if extended else 'raw'}.png')

    # Performing the significance test
    for key1 in dic_performace.keys():
        for key2 in dic_performace.keys():
            if key1 != key2:
                p_value = significance_test(dic_performace[key1], dic_performace[key2], random_seed=random_state)
                df_t_test = df_t_test._append({'source': data_source, 'model': 'LSTM-paper-keras', 'dataset1' : key1, 'dataset2' : key2, 'p-value' : p_value}, ignore_index=True)

# Aggregating the results and taking the average
result = result.groupby(['source', 'model', 'dataset', 'extended'], as_index=False).mean()

# Significance test
df_t_test = df_t_test.groupby(['source', 'model', 'dataset1', 'dataset2'], as_index=False).mean()

# Saving the results
result.to_csv(f'{output_dir}/results.csv', index=False)
df_t_test.to_csv(f'{output_dir}/significance-test.csv', index=False)

print(f"Time taken: {time.time() - start} seconds")