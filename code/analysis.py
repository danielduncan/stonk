# analyses stock data by observing historical data and fitting a function to the dataset using a PyTorch neural network

# imports retrieved data
import data
# imports for neural network
import torch
import torch.nn as nn
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time

# neural network initialise
dtype = torch.float
device = torch.device("cpu")

x_train = torch.from_numpy(data.x_train).type(torch.Tensor)
x_test = torch.from_numpy(data.x_test).type(torch.Tensor)

# lstm
y_train_lstm = torch.from_numpy(data.y_train).type(torch.Tensor)
y_test_lstm = torch.from_numpy(data.y_test).type(torch.Tensor)

# gru
'''
y_train_gru = torch.from_numpy(data.y_train).type(torch.Tensor)
y_test_gru = torch.from_numpy(data.y_test).type(torch.Tensor)
'''

# common features of both networks
input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100 # use 100

# LSTM (slower, better predictions)
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

'''
# GRU (faster, similar predictions)
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out

model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
'''

hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []

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

preditctionSet = data.scaler.inverse_transform(y_train_pred.detach().numpy())
labelSet = data.scaler.inverse_transform(y_train_lstm.detach().numpy())

print("predictions:\n" + str(preditctionSet[-1]))
print("labels:\n" + str(labelSet[-1]))