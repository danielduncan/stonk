# analyses stock data by observing historical data and fitting a function to the dataset using a PyTorch neural network

# imports retrieved data
import data
# imports for neural network
import torch
import math

# neural network initialise
dtype = torch.float
device = torch.device("cpu")

# generates random data as an input (to be fixed later)
x = torch.linspace(-math.pi, math.pi, 36, device=device, dtype=dtype)
# inputs the historical closing prices of asset
y = torch.from_numpy(data.historical_close_prices)

# this part isn't mine... source is unknown
# randomised network nodes
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(len(data.historical_close_prices)):
    # forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # backprop(ogate?) to compute the gradients of a, b, c and d with respect to loss
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

# sets the input of the price movement function to 9999 (maybe seconds?) from beginning of dataset
datetime = 9999
z = a.item() * datetime + b.item() * datetime + \
    c.item() * datetime + d.item() * datetime

# outputs the predicted price of the stock at the given time
print(z)
