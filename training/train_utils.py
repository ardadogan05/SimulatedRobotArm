from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training.torch_utils import get_torch_device


def train_neural_ik(
    model,
    dataset_path,
    model_path,
    losses_path,
    n_epochs=100,
    batch_size=128,
    learning_rate=1e-3,
):
    #Allows training to happen on GPU if possible, otherwise fallback to CPU.
    #Gives faster speeds (not that important for these small NNs)
    device = get_torch_device()

    print("Using device:", device)

    with np.load(dataset_path) as data:
        X_train = data["X_train"]
        Y_train = data["Y_train"]
        X_val = data["X_val"]
        Y_val = data["Y_val"]

    #numpy arrays have to be converted to pytorch tensors.
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)

    #wrapping training data in DataLoader.
    #This is to feed training data to nn in mini-batches instead of all at once
    train_dataset = TensorDataset(X_train, Y_train) #pairs input with correct target
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, #number of examples processed at once
        shuffle=True, #shuffles order each epoch
    )

    #Moves the model to the selected device
    model = model.to(device)

    # Loss function: function used for optimization later, as described in PDF.
    # Mean square error between the predicted output and expected target
    loss_fn = nn.MSELoss()

    #Optimizer. Responsible for adjusting weights, using the selected learning rate
    #(too large -> jump around or unstable, too small -> slow)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # "Adam" seems to be a form of gradient descent with adjustment of step size
    # based on recent behavior. Should result in more effective training

    #Moves validation data to selected device
    X_val = X_val.to(device)
    Y_val = Y_val.to(device)

    #epoch is one full pass through all the training samples.
    #we go through it n_epochs times.

    #To save the losses
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        model.train() #sets nn to training mode

        #accumulates total training loss for each epoch.
        #Tells us how well current weights and biases perform on training dataset
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            #Moves batch to selected device
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            #forward pass
            predictions = model(X_batch)

            loss = loss_fn(predictions, Y_batch)

            #Backpropagation based on the loss function
            #Forgets gradients from previous batch (pytorch accumulates by default)
            optimizer.zero_grad()
            loss.backward() #the actual backpropagation. Calculates gradients
            #updates weights and biases, should move in a direction that reduces loss
            #(towards local minima)
            optimizer.step()

            #.item to convert to normal python number.
            #shape[0] gives number of samples in batch
            train_loss += loss.item() * X_batch.shape[0]

        train_loss /= len(train_dataset) #gives average training loss for the epoch

        #validation
        #sets nn to evaluation mode.
        #Used to measure performance on data not used for training.
        model.eval()
        with torch.no_grad(): #turns off gradient tracking, this part is strictly eval
            val_predictions = model(X_val)
            val_loss = loss_fn(val_predictions, Y_val).item()

        print(
            f"Epoch {epoch + 1:03d} | "
            f"train loss: {train_loss:.6f} | "
            f"val loss: {val_loss:.6f}"
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    #Save model
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    # Save loss history for plotting later
    losses_path = Path(losses_path)
    losses_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        losses_path,
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
    )

    return train_losses, val_losses
