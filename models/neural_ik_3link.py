import torch
from torch import nn


#defines a neural network class, inherits from PyTorch nn base class
class NeuralIK3Link(nn.Module):
    def __init__(self): #defines constructor for class
        super().__init__() #calls the constructor of nn.Module

        self.network = nn.Sequential(
            #Creates sequence of layers
            nn.Linear(5, 128), #computes Z1 as described in 3 link NN pdf
            nn.ReLU(), #H1
            nn.Linear(128, 128), #Z2
            nn.ReLU(), #H2
            nn.Linear(128, 3), #Y_hat
        ) #Full network is now stored in self.network

    #Runs input through the entire network.
    #Returns predicted angles.
    def forward(self, x):
        return self.network(x)
