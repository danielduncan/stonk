# import yahoo finance for real time market data
import yfinance as yf
# import pandas and numpy for data analysis
import pandas as pd
import numpy as np
# import torch for neural network
import torch


def obtain_data(ticker, first, last):
    # yfinance usage is ticker = yf.Ticker("TICKER")
    # refer to yahoo finance for difficult tickers
    class stock:
    
        data = yf.Ticker(ticker)

        # date usage is YYYY-MM-DD
        raw_historical = yf.download(ticker, start=first, end=last, auto_adjust = True)
        numpy_historical = raw_historical.to_numpy()

        dates = raw_historical.index[:]
        historical_close_prices = numpy_historical[:,3]

        global date_close_frame

        date_close_frame = pd.DataFrame({'Date':dates, 'Close Price':historical_close_prices})

    # data is stored in a dictionary
    # print(stock.data.info["regularMarketPrice"]) # debug printout

# obtain_data(input(), input(), input())
obtain_data('NIO', '2020-01-01', '2021-01-01')

print(date_close_frame)

closes = np.array(date_close_frame['Close Price'])