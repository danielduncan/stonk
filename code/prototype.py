# important imports
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import math

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import *

import threading
import time


# currently set to trade AMD stock
raw_historical = yf.download('AMD', period='ytd', auto_adjust=True)
numpy_historical = raw_historical.to_numpy()
date = raw_historical.index[:]
historical_close_prices = numpy_historical[:, 3]

# neural network initialise
dtype = torch.float
device = torch.device("cpu")

# generates random shit for input data
x = torch.linspace(-math.pi, math.pi, 35, device=device, dtype=dtype)
# historical closing prices of AMD
y = torch.from_numpy(historical_close_prices)

# this isn't mine can't remember the source though
# randomised nodes
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(len(historical_close_prices)):
    # forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

# outputs the price movement function formula
print(
    f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')

# 9999n from beginning of dataset
datetime = 9999
z = a.item() * datetime + b.item() * datetime + \
           c.item() * datetime + d.item() * datetime
print(z)


# further important imports


# interactive brokers shit
class IBapi(EWrapper, EClient):
	def __init__(self):
		EClient.__init__(self, self)

	def nextValidId(self, orderId: int):
		super().nextValidId(orderId)
		self.nextorderId = orderId
		print('The next valid order id is: ', self.nextorderId)

	def orderStatus(self, orderId, status, filled, remaining, avgFullPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
		print('orderStatus - orderid:', orderId, 'status:', status, 'filled',
		      filled, 'remaining', remaining, 'lastFillPrice', lastFillPrice)

	def openOrder(self, orderId, contract, order, orderState):
		print('openOrder id:', orderId, contract.symbol, contract.secType, '@', contract.exchange,
		      ':', order.action, order.orderType, order.totalQuantity, orderState.status)

	def execDetails(self, reqId, contract, execution):
		print('Order Executed: ', reqId, contract.symbol, contract.secType, contract.currency,
		      execution.execId, execution.orderId, execution.shares, execution.lastLiquidity)


def run_loop():
	app.run()

# create the order


def stockOrder(symbol):
	contract = Contract()
	contract.symbol = 'AMD'
	contract.secType = 'STK'; contract.currency = 'USD' # fuck this but unindent is being a pain
	contract.exchange = 'ISLAND'
	return contract

app = IBapi()
app.connect('127.0.0.1', 4002, 420)

app.nextorderId = None

# start the socket in a thread
api_thread = threading.Thread(target=run_loop, daemon=True)
api_thread.start()

# check if the API is connected via orderid
while True:
	if isinstance(app.nextorderId, int):
		print('connected')
		break
	else:
		print('waiting for connection')
		time.sleep(1)

# create order object
order = Order()
order.action = 'BUY'
order.totalQuantity = 1
order.orderType = 'LMT'
order.lmtPrice = yf.Ticker('AMD').info["regularMarketPrice"]

# determines if stock is worth buying by comparing current price with predicted price
if (z > yf.Ticker('AMD').info["regularMarketPrice"]):
    app.placeOrder(app.nextorderId, stockOrder('AMD'), order)
else:
    print('no action')

'''
# Cancel order
print('cancelling order')
app.cancelOrder(app.nextorderId)
'''
time.sleep(3)

app.disconnect()
