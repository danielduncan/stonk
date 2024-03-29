# retrieves important asset data using the yfinance library
# for webscraping the required data from Yahoo finance
import yfinance as yf
# for processing data
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# retrieves the assets ticker
def retrieveData(ticker):
    # seperates historical data into unique date and closing price arrays
    raw_historical = yf.download(ticker, period='max', auto_adjust=True)
    numpy_historical = raw_historical.to_numpy()
    date = raw_historical.index[:]
    historical_close_prices = numpy_historical[:, 3]
    data = pd.DataFrame(index=date, columns=['Close'])

    # hacky method of formatting data
    for i in range(len(historical_close_prices)):
        data.loc[date[i], 'Close'] = historical_close_prices[i]

    # definitely shouldn't be global... fix before release
    global price
    price = data[['Close']]

    # reshapes dataset within bounds -1 and 1 for the neural network
    global scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1, 1))
    return scaler


# returns current market price of an asset given its ticker
def mktPrice(ticker):
    # current market price of asset
    mktPrice = yf.Ticker(ticker).info["regularMarketPrice"]
    return mktPrice


# splits data into training and testing sets
def dataSplit(inData, lookback):

    raw = inData.to_numpy()  # convert to numpy array
    clean = []

    # create all possible sequences of length seq_len
    for index in range(len(raw) - lookback):
        clean.append(raw[index: index + lookback])

    clean = np.array(clean)
    test_set_size = int(np.round(0.2*clean.shape[0]))
    train_set_size = clean.shape[0] - (test_set_size)

    x_train = clean[:train_set_size, :-1]
    y_train = clean[:train_set_size, -1]

    x_test = clean[train_set_size:, :-1]
    y_test = clean[train_set_size:, -1]

    return [x_train, y_train, x_test, y_test]


# lookback is for sliding window method
lookback = 365


# forms the dataset for the neural network given a ticker
def formSet(ticker):
    retrieveData(ticker)
    x_train, y_train, x_test, y_test = dataSplit(price, lookback)
    return x_train, y_train, x_test, y_test
