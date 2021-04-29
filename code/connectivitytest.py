# for testing connectivity with the Interactive Brokers servers
# not mine, provided by Interactive Brokers

from ibapi.client import EClient
from ibapi.wrapper import EWrapper


class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)


app = IBapi()
app.connect('127.0.0.1', 4002, 42)
app.run()