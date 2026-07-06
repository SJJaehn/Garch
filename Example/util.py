import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


"""
Reshaping the data to fit the LSTM model
"""
def create_sequence(X, y, perf, lookback):
    Xs = []
    ys = []
    perfs = []
    for i in range(len(X) - lookback):
        v = X.iloc[i:i + lookback].values
        Xs.append(v)
        ys.append(y.iloc[i + lookback - 1])
        perfs.append(perf.iloc[i + lookback - 1])
    return np.array(Xs), np.array(ys), np.array(perfs)


"""
This function calculates the threshold for the target variable while taking into account the recall range.
The threshold is chosen by the highest precision in the valid recall range.
"""
def find_threshold(y_pred, y_true, min_recall = 0, max_recall = 1):
    precision, recall, threshold = precision_recall_curve(y_true, y_pred)

    valid_index = np.where((recall <= max_recall) & (recall >= min_recall)) # Check for the valid recall range
    valid_threshold = threshold[valid_index]
    valid_precision = precision[valid_index]
    
    if len(valid_precision) == 0: # If there is no valid result, return 1 as the threshold
            return 1
    best_threshold = valid_threshold[np.argmax(valid_precision)] # Find the highest precision in the valid range
    
    return best_threshold

"""
This function finds the best threshold for the average return.
It iterates through all possible thresholds and calculates the average return for each threshold.
The threshold with the highest average return is returned.

This was part in my expose as an optional addition, but ended up not being used due to scope limitations.
"""
def find_threshold_average_return(y_pred, perf):
    best_threshold = 1 # Setting the best threshold to 1, 1 would mean that the model would not make any trades
    best_average_return = 0 # Setting the best average return to 0

    y_pred = y_pred.flatten()

    temp = pd.Series(y_pred).sort_values().rolling(window=2).mean()
    for threshold in temp:
        y_temp = (y_pred >= threshold).astype(int)
        average_return = np.mean(perf * y_temp) # Calculate the average return for the current threshold
        if average_return > best_average_return:
            best_average_return = average_return
            best_threshold = threshold

    return best_threshold


"""
This function calculates the sharpe ratio, precision, recall and average return
I use the equation for the sharpe ratio and average return from the paper.
The equation for the sharpe ratio was provided by the author of the paper on request.

This function also plots the cumulative returns of the strategy and the buy-and-hold strategy.
"""
def calculate_metrics(y_true, y_pred, perf, threshold):
    y_pred = (y_pred >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=np.nan)
    recall = recall_score(y_true, y_pred, zero_division=np.nan)

    returns = perf * y_pred.flatten()
    cum_returns = np.cumprod(returns + 1) # Cumulative returns using 1 as the starting point

    trade_returns = returns[y_pred.flatten() == 1] # Removing the periods without a trade

    average_return = (cum_returns[-1] - 1) / len(trade_returns) # Average return when a trade is executed

    # Sharpe ratio assuming a 0% risk-free rate
    num_trades = len(trade_returns)
    num_periods_total = len(returns)
    num_periods_year = 365 * 24

    if num_trades == 0: # Avoid division by zero
        sharpe_ratio = np.nan
        
        recall = np.nan # Set recall to 0 when no trade is made so that every value is set to nan
    else:
        annulized_return = (cum_returns[-1]) ** (num_periods_year/len(returns)) - 1 # Annualized return
        sharpe_ratio = annulized_return / (np.std(returns) * np.sqrt(num_periods_year) ) # Sharpe ratio
        #sharpe_ratio = np.sqrt(num_periods_year) * np.mean(returns) / np.std(returns)

    # Plotting the cumulative returns
    buy_and_hold = np.cumprod(perf + 1) # Cumulative returns of the buy-and-hold strategy
    

    fig = plt.figure(figsize=(12, 6))
    plt.plot(cum_returns, label='Strategy Returns')
    plt.plot(buy_and_hold, label='Buy and Hold Returns')
    plt.title('Cumulative Returns')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

    return sharpe_ratio, precision, recall, average_return, fig

# This function follows the description given in the paper. For this replication, I used the cumulative returns of the strategy (see calculate_metrics function) to plot the strategy against the bitcoin price.
def simulate_strategy(y_pred_long, perf):
    strategy = [0] # Setting 0 as the start value
    
    # temp variable to keep track on the value of the currently held position
    pos_long = 0
    gain_long = 0
    

    for i in range(len(y_pred_long)): # Going through all the results
        
        if y_pred_long[i] == 1: # If you need to buy 
            

            # Adding $1 to the position every time a buy signal is given but also keep running the bitcoin from the periode before
            if (pos_long == 0):
                pos_long = 1
            

            # Calculate the gain/loss of the position in this periode
            gain_long = pos_long * perf[i]
            pos_long += gain_long # Add the gain/loss to the position
        else:
            pos_long = 0 # Selling the position
            gain_long = 0

        strategy.append(strategy[-1] + gain_long) # Adding the gain/loss to the money made by the strategy

    plt.figure()
    plt.plot(strategy, label="Strategy")
    return strategy



def get_cyclical_features(df:pd.DataFrame):
    # Calculating cycical features
    # Extract hour of day and day of week
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    # Create cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    df.drop(columns=['hour', 'day_of_week'], inplace=True) # Dropping the temporary columns

    return df

def get_candle_features(df:pd.DataFrame):
    # Calculating the candle features
    df['body'] = abs(df['close'] - df['open'])
    df['color'] = (df['close'] >= df['open']).astype(int)
    df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']

    return df

"""
This function calculates the technical indicators for the given dataframe.
"""
def get_indicators(df:pd.DataFrame):
    
    df = calculate_sma(df, 15)
    df = calculate_sma(df, 20)
    df = calculate_sma(df, 25)
    df = calculate_sma(df, 30)
    
    df = calculate_rsi(df, 15)
    df = calculate_rsi(df, 20)
    df = calculate_rsi(df, 25)
    df = calculate_rsi(df, 30)

    df = calculate_williams_r(df)

    df = calculate_macd(df)

    df = calculate_mfi(df)

    df = calculate_stochastic_oscillator(df)


    return df





"""
Functions for calculating technical indicators
"""
def calculate_sma(df:pd.DataFrame, window:int): # Calculating the SMA
    df[f"sma_{window}"] = df['close'].rolling(window=window).mean()
    return df


def calculate_rsi(df:pd.DataFrame, window:int=15): # Calculating RSI using the EMA-based method
    delta = df['close'].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(com=window-1, adjust=False).mean()
    avg_loss = loss.ewm(com=window-1, adjust=False).mean()

    rs = avg_gain / avg_loss
    df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
    return df


def calculate_williams_r(df:pd.DataFrame, window:int=14): # Calculating Williams %R
    high = df['high'].rolling(window=window).max()
    low = df['low'].rolling(window=window).min()
    close = df['close']

    williams_r = -100 * (high - close) / (high - low)
    df[f"williams_r_{window}"] = williams_r
    return df

def calculate_macd(df:pd.DataFrame, short_window:int=12, long_window:int=26, signal_window=9): # Calculating MACD
    short_ema = df['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    #macd_ema = macd.ewm(span=signal_window, adjust=False).mean()
    #histogram = macd - macd_ema

    df[f"macd_{short_window}_{long_window}"] = macd # Using this because it fits the min/max values in table 3
    return df


def calculate_mfi(df:pd.DataFrame, window:int=14): # Calculating Money Flow Index
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    positive_flow = np.where(typical_price.diff() > 0, money_flow, 0)
    negative_flow = np.where(typical_price.diff() < 0, money_flow, 0)
    
    positive_mf = pd.Series(positive_flow).rolling(window=window).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=window).sum()
    
    mfi = 100 * (positive_mf / (positive_mf + negative_mf))
    
    df[f"mfi_{window}"] = mfi.values
    return df


def calculate_stochastic_oscillator(df:pd.DataFrame, window:int=14, smooth_d:int=3): # Calculating SO
    high = df['high'].rolling(window=window).max()
    low = df['low'].rolling(window=window).min()
    close = df['close']

    k = 100 * ((close - low) / (high - low))
    # d = k.rolling(window=smooth_d).mean()

    df[f"so_{window}"] = k # Using k as it fits the values in table 3
    return df


"""
This function performs a significance test using the block bootstrap method.
This function follows the description given in the paper.
The Welch t-test is used because there is no mention in the paper of the type of t-test used.
"""
def significance_test(ret_a, ret_b, n_iter:int = 10000, random_seed:int = 42):
    # Make sure both arrays are numpy arrays and the same length
    ret_a = np.array(ret_a)
    ret_b = np.array(ret_b)

    # Calculating the block size
    n = len(ret_a)
    k = max(int(n ** (1/3)), 1) # Using ^1/3 (min 1) as the block size following the paper

    b = int(n / k) # Number of blocks

    # Cutting of the tail of the arrays to avoid blocks with less than k elements
    ret_a = ret_a[:b*k]
    ret_b = ret_b[:b*k]

    # Splitting the returns into n/k blocks.
    blocks_a = np.array([ret_a[i:i + k] for i in range(0, len(ret_a), k)])
    blocks_b = np.array([ret_b[i:i + k] for i in range(0, len(ret_b), k)])

    observed_diff = ret_a.mean() - ret_b.mean() # Calculate the difference between the means of the two blocks
    observed_t = observed_diff / np.sqrt((ret_a.var(ddof=1) / len(ret_a)) + (ret_b.var(ddof=1) / len(ret_b))) # Calculate the t-statistic, using ddof=1 to get the sample variance

    # Setting the random seed for reproducibility
    rng = np.random.default_rng(random_seed)
    bootstrap_ts = np.empty(n_iter)

    # Loop n_iter times to create the bootstrap samples
    for i in range(n_iter):
        # Selecting random blocks from both samples, allowing for replacement
        idx = rng.integers(0, b, size=b)
        # Using the same block indices for both samples
        sample1 = blocks_a[idx].ravel() # ravel() to flatten the array
        sample2 = blocks_b[idx].ravel()
        # Calculate the difference between the means of the two samples
        bootstrap_diff = sample1.mean() - sample2.mean()
        # Calculate the t-statistic for the bootstrap sample
        bootstrap_ts[i] = bootstrap_diff / np.sqrt((sample1.var(ddof=1) / len(sample1)) + (sample2.var(ddof=1) / len(sample2)))

    # Calculate the p-value (two sided t-test)
    p_value = np.mean(np.abs(bootstrap_ts) >= np.abs(observed_t))

    return p_value