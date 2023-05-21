#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 18:42:19 2023

@author: ouj
"""

import torch
import torch.nn as nn
from timeit import default_timer as timer
import scipy.io
import numpy as np
import scipy.io as sio



# Define the neural network class
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(3, 3)  # input layer to hidden layer
        self.fc2 = nn.Linear(3, 1)  # hidden layer to output layer
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # apply ReLU activation to hidden layer
        x = self.fc2(x)  # output layer with no activation function
        return x

# Create an instance of the neural network
net = NeuralNet()


# Define the loss function and optimizer
criterion = nn.MSELoss()
# optimizer = torch.optim.LBFGS(net.parameters(), lr=0.001)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
# Create some training data
# inputs = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]])
# targets = torch.tensor([[3.0], [4.0], [5.0], [6.0]])

# load data
training_data = scipy.io.loadmat('training_40.mat')
testing_data = scipy.io.loadmat('testing.mat')
inputs = torch.tensor(np.float32(training_data["Xc_black"]))
targets = torch.tensor(np.float32(training_data["Xw_black"][:,0].reshape(40,1)))
test_inputs = torch.tensor(np.float32(testing_data["Xc_black_test"][0,:]))
test_targets = torch.tensor(np.float32(testing_data["Xw_black_test"][:,0].reshape(10,1)))

# Train the neural network
for epoch in range(1000):
    # Forward pass
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    def closure():
        return loss
    # def closure():
    #       if torch.is_grad_enabled():
    #         optimizer.zero_grad()
    #       # output = net(inputs)
    #       if loss.requires_grad:
    #         loss.backward()
    #       return loss
    optimizer.step()
    
    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        

# t_input = test_inputs[0,:]
time_ANN = np.zeros(100)
for i in range(100):
    start = timer()
    # Test the neural network
    # test_inputs = torch.tensor([[5.0, 6.0, 7.0]])
    test_outputs = net(test_inputs)
    end = timer()
    time_ANN[i] = end - start
filename = 'time.mat'
# sio.savemat(filename, {'time':time_ANN})
print(end-start)
# diffence =test_targets.detach().numpy()-test_outputs.detach().numpy()


# # Print the output tensor
# print(test_outputs)