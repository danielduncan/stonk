# interface between the code and the Interactive Brokers API

# imports analysis and data
import analysis
import data

# Interactive Brokers imports
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import *

# time and threading for optimisation
import time
import threading

# interactive brokers class
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
    contract.symbol = data.ticker
    contract.secType = 'STK'
    contract.currency = 'USD'  # fuck this but unindent is being a pain
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
order.lmtPrice = data.mktPrice

# determines if stock is worth buying by comparing current price with predicted price
if (analysis.z > data.mktPrice):
    app.placeOrder(app.nextorderId, stockOrder(data.ticker), order)
else:
    print('no action')

time.sleep(3)

app.disconnect()
