# analyses stock data by observing historical data and fitting a function to the dataset using a PyTorch neural network

# imports retrieved data
import data
# imports for neural network
import torch
import torch.nn as nn
import math
import numpy as np
import time

# neural network initialise
dtype = torch.float
device = torch.device("cpu")

x_train = torch.from_numpy(data.x_train).type(torch.Tensor)
x_test = torch.from_numpy(data.x_test).type(torch.Tensor)
y_train_lstm = torch.from_numpy(data.y_train).type(torch.Tensor)
y_test_lstm = torch.from_numpy(data.y_test).type(torch.Tensor)
y_train_gru = torch.from_numpy(data.y_train).type(torch.Tensor)
y_test_gru = torch.from_numpy(data.y_test).type(torch.Tensor)

# common features of nn
input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100

# LSTM
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []

print(num_epochs)

for t in range(num_epochs):
    y_train_pred = model(x_train)

    loss = criterion(y_train_pred, y_train_lstm)
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
training_time = time.time()-start_time
print("Training time: {}".format(training_time))


'''
# effectively a random backprop nn

# generates random data as an input (to be fixed later)
x = torch.linspace(-math.pi, math.pi, 41, device=device, dtype=dtype)
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

    # backpropagate to compute the gradients of a, b, c and d with respect to loss
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
currtime = 9999

# predicted polynomial
prediction = a.item() * currtime + b.item() * currtime + \
    c.item() * currtime + d.item() * currtime

# outputs the predicted price of the stock at the given time
print(prediction)
'''