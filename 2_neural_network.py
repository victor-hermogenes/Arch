import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.rand(input_size, hidden_size)
        self.weights2 - np.random.rand(hidden_size, output_size)
        self.bias1 = np.random.rand(hidden_size)
        self.bias2 = np.random.rand(output_size)


    def forward(self, x):
        self.input = x
        self.hidden = sigmoid(np.dot(x, self.weights1) + self.bias1)
        self.output = sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)
        return self.output
    

    def backward(self, y_true, learning_rate):
        output_error = y_true - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        hidden_error = np.dot(output_delta, self.weights2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden)

        self.weights2 += np.dot(self.hidden.T, output_delta) * learning_rate
        self.weights1 += np.dot(self.input.T, hidden_delta) * learning_rate
        self.bias2 += np.sum(output_delta, axis=0) * learning_rate
        self.bias1 += np.sum(hidden_delta, axis=0) * learning_rate