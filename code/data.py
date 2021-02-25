# retrieves important asset data using the yfinance library

# imports the yfinance library used for webscraping the required data from Yahoo finance
import yfinance as yf

# retrieves the assets ticker
ticker = input("Ticker to be traded: ")

# seperates historical data into unique date and closing price arrays
raw_historical = yf.download(ticker, period='ytd', auto_adjust=True)
numpy_historical = raw_historical.to_numpy()
date = raw_historical.index[:]
historical_close_prices = numpy_historical[:, 3]

# current market price of asset
mktPrice = yf.Ticker(ticker).info["regularMarketPrice"]
