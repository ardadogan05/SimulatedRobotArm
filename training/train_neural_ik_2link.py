import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from models.neural_ik_2link import NeuralIK2Link

def main():
    data = np.load("data/processed/ik_2link_dataset.npz")

    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_val = data["X_val"]
    Y_val = data["Y_val"]

    #numpy arrays have to be converted to pytorch tensors.
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)

    #wrapping training data in DataLoader. This is to feed training data to nn in mini-batches instead of all at once
    train_dataset = TensorDataset(X_train, Y_train) #pairs input with correct target
    train_loader = DataLoader(
        train_dataset, 
        batch_size= 128, #128 examples at once
        shuffle=True #shuffles order each epoch
    )
    #creating model from other file, calls on its constructor
    model = NeuralIK2Link()

    # Loss function: function used for optimization later, as described in PDF
    loss_fn = nn.MSELoss()

    #Optimizer. Responsible for adjusting weights, using learning rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #(too large -> jump around or unstable, too small -> slow)
    # "Adam" seems to be a form of gradient descent with smart adjustment of step size based on recent behavior. Should result in more effective training

    n_epochs = 100 #epoch is one full pass through the 40000 training samples, we go through it 100 times.

    for epoch in range(n_epochs):
        model.train() #sets nn to training mode

        train_loss = 0.0 #accumulates total training loss for each epoch. CONTINUE HERE
        for X_batch, Y_batch in train_loader:
            #forward pass
            predictions = model(X_batch)

            loss = loss_fn(predictions, Y_batch)

            #Backpropogation based on Loss function
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
