# retrieves important asset data using the yfinance library

# imports the yfinance library used for webscraping the required data from Yahoo finance
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

# retrieves the assets ticker
ticker = input("Ticker to be traded: ")

# seperates historical data into unique date and closing price arrays
raw_historical = yf.download(ticker, period='max', auto_adjust=True)
numpy_historical = raw_historical.to_numpy()
date = raw_historical.index[:]
historical_close_prices = numpy_historical[:, 3]
data = pd.DataFrame(index=date, columns=['Close'])

# this is a really hacky solution
for i in range(len(historical_close_prices)):
    data.loc[date[i], 'Close'] = historical_close_prices[i]

print(data)

# current market price of asset
mktPrice = yf.Ticker(ticker).info["regularMarketPrice"]

price = data[['Close']]

# reshapes dataset within bounds -1 and 1 for the neural network
scaler = MinMaxScaler(feature_range=(-1, 1))
price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))
print(price['Close'])

# lookback is for sliding window method
def dataSplit(inData, lookback):

    raw = inData.to_numpy() # convert to numpy array
    clean = []
    
    # create all possible sequences of length seq_len
    for index in range(len(raw) - lookback): 
        clean.append(raw[index: index + lookback])

    clean = np.array(clean)
    test_set_size = int(np.round(0.2*clean.shape[0]))
    train_set_size = clean.shape[0] - (test_set_size)

    x_train = clean[:train_set_size,:-1]
    y_train = clean[:train_set_size,-1]
    
    x_test = clean[train_set_size:,:-1]
    y_test = clean[train_set_size:,-1]
    
    return [x_train, y_train, x_test, y_test]

lookback = 365

x_train, y_train, x_test, y_test = dataSplit(price, lookback)