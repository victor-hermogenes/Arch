from neural_network import NeuralNetwork as nw
import numpy as np
import data_processing as dp

# Load and process data

file_path = input("Paste your csv path: ") # Specify the path to your data file
data = dp.load_data(file_path)
data = dp.normalize_data(data)

# Assuming last column is the label
X = data[:, :-1]
y = data[:, -1].reshape(-1, 1)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = dp.train_test_split(X, y, test_size=0.2)

# Hyperparameters
input_size = x_train.shape[1]
hidden_size = 3
output_size = 1
learning_rate = 0.1
epochs = 10000

# Initialize Neural Network
nn = nw(input_size, hidden_size, output_size)

# Training loop
for epoch in range(epochs):
    # Forward Pass
    output = nn.forward(x_train)

    # Backward Pass and Weight Update
    nn.backward(y_train, learning_rate)

    # Pint Loss Every 1000 Epochs
    if epoch % 1000 == 0:
        loss = np.mean(np.square(y_train - output))
        print(f'Epch {epoch}, Loss: {loss}')

    
# Testing the Neural Network
test_output = nn.forward(x_test)
test_loss = np.mean(np.square(y_test - test_output))
print(f'Test Loss: {test_loss}')