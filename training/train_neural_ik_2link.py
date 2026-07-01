from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from models.neural_ik_2link import NeuralIK2Link

def main():

    #Allows training to happen on GPU if possible, otherwise fallback to CPU. Gives faster speeds (not that important for this small NN)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)


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
    model = NeuralIK2Link().to(device)

    # Loss function: function used for optimization later, as described in PDF. Mean square error between predicted and actual value
    loss_fn = nn.MSELoss()

    #Optimizer. Responsible for adjusting weights, using learning rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #(too large -> jump around or unstable, too small -> slow)
    # "Adam" seems to be a form of gradient descent with adjustment of step size based on recent behavior. Should result in more effective training

    #Moves validation data to selected device
    X_val = X_val.to(device)
    Y_val = Y_val.to(device)


    n_epochs = 100 #epoch is one full pass through the 40000 training samples, we go through it 100 times.

    #To save the losses
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        model.train() #sets nn to training mode

        train_loss = 0.0 #accumulates total training loss for each epoch. Tells us how well current weights and biases perform on training dataset
        for X_batch, Y_batch in train_loader:
            #Moves batch to selected device
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            #forward pass
            predictions = model(X_batch)

            loss = loss_fn(predictions, Y_batch)

            #Backpropagation based on the loss function
            optimizer.zero_grad() #Forgets gradients from previous batch (pytorch accumulates by default)
            loss.backward() #the actual backpropagation. Calculates gradients
            optimizer.step() #updates weights and biases, should move in a direction that reduces loss (towards local minima)

            train_loss += loss.item() * X_batch.shape[0] #.item to convert to normal python number, shape[0] gives number of samples in batch

        train_loss /= len(train_dataset) #gives average training loss for the epoch

        #validation
        model.eval() #sets nn to evaluation mode. Used to measure performance on data not used for training.
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
    Path("models/saved").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "models/saved/neural_ik_2link.pt")

    # Save loss history for plotting later
    Path("results/phase4").mkdir(parents=True, exist_ok=True)

    np.savez(
        "results/phase4/neural_ik_2link_losses.npz",
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
    )

if __name__ == "__main__":
    main()