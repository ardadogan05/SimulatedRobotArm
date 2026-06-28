from data.generate_2link_dataset import generate_2link_dataset
import numpy as np
import matplotlib.pyplot as plt

def plot_dataset(X):
    plt.scatter(X[:, 0], X[:, 1], s=1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2-link IK dataset target positions")
    plt.show()

def split_dataset(X,Y): #Will use 40k for training, 5k for val and 5k for test.
    indices = np.random.permutation(len(X)) #random shuffle of indices

    train_end = int(0.8 * len(X)) #80% of data goes to training
    val_end = int(0.9 * len(X)) #10% to validation

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test



def main():
    # Generate dataset
    X, Y = generate_2link_dataset(50_000)

    # Split dataset
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_dataset(X, Y)

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_val shape:", X_val.shape)
    print("Y_val shape:", Y_val.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)

    print("x range:", X[:, 0].min(), X[:, 0].max())
    print("y range:", X[:, 1].min(), X[:, 1].max())

    print("theta1 range:", Y[:, 0].min(), Y[:, 0].max())
    print("theta2 range:", Y[:, 1].min(), Y[:, 1].max())

    # Save split dataset
    np.savez(
        "data/processed/ik_2link_dataset.npz",
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        X_test=X_test,
        Y_test=Y_test,
    )

    # Plot full generated dataset
    plot_dataset(X)

if __name__ == "__main__":
    main()