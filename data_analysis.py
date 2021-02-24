# import yahoo finance for real time market data
import yfinance as yf
# import pandas and numpy for data analysis
import pandas as pd
import numpy as np
# import torch for neural network
import torch


def obtain_data(ticker):
    # yfinance usage is ticker = yf.Ticker("TICKER")
    # refer to yahoo finance for difficult tickers
    stock = yf.Ticker(ticker)

    # data is stored in a dictionary
    print(stock.info["regularMarketPrice"])


obtain_data(input())