import data_processing as dp
from neural_network import NeuralNetwork as nw
import numpy as np
import utils


def main():
    # Load and preprocess data
    file_path = input("Give your excel file path")
    data = dp.load_data(file_path)
    data = dp.normalize_data(data)

    # Assuming the last column is the label
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = dp.train_test_split(X, y, test_size=0.2)

    # hyperparameters
    input_size = X_train.shape[1]
    hidden_size = 3
    output_size = 1
    learning_rate = 0.1
    epochs = 10000

    # Initialize Neural Networkd
    nn = nw(input_size, hidden_size, output_size)

    # Training Loop
    losses = []
    for epoch in range(epochs):
        #Forward Pass
        output = nn.forward(X_train)


        # Backward pass and Weight Update
        nn.backward(y_train, learning_rate)

        # Calculate and log loss
        loss = np.mean(np.square(y_train - output))
        losses.appen(loss)
        if epoch % 1000 == 0:
            utils.plot_loss(epoch, loss)

    # Plot training loss
    utils.plot_loss(losses)

    # Testing the Neural Networkd
    test_output = nn.forward(X_test)
    test_loss = np.mean(np.square(y_test - test_output))
    print( f'Test Loss {test_loss}')

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f'Error: {e}')