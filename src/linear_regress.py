import imp
from pickletools import optimize
from turtle import forward


import torch
import torch.nn as nn
import numpy as np


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        out = self.linear(x)
        return out


def train():
    x_values = [i for i in range(11)]
    x_train = np.array(x_values, dtype=np.float32)
    x_train = x_train.reshape(-1, 1)
    y_values = [2*x + 1 for x in x_values]
    y_train = np.array(y_values, dtype=np.float32)
    y_train = y_train.reshape(-1, 1)

    input_dim = 1
    output_dim = 1
    model = LinearRegressionModel(input_dim, output_dim)
    epochs = 300
    lr = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        X = torch.from_numpy(x_train)
        Y = torch.from_numpy(y_train)
        optimizer.zero_grad()
        Y_hat = model(X)
        loss = criterion(Y_hat, Y)
        loss.backward()
        optimizer.step()
        if epoch % 30 == 0:
            print('epoch {}, loss {}'.format(epoch, loss.item()))
    
    return model.state_dict()
