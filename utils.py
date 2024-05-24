import numpy as np
import matplotlib.pyplot as plt


def log_progress(epoch, loss):
    """
    Log the training progress.
    Args:
        epoch (int): the current epoch.
        loss (float): The current loss.
    """
    print(f'Epoch {epoch}, Loss:{loss}')


def plot_loss(losses):
    """"
    Plot the training loss over epochs.
    Args:
        losses (list): List of loss values over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.show()


def save_model(model, file_path):
    """
    Save the model weights and biases.
    Args:
        model (NeuralNetwork): The neural network model.
        file_path (str): Path to save the model.
    """
    np.savez(file_path, weight1s=model.weights1, weights2=model.weights2, bias1=model.bias1, bias2=model.bias2)


def load_model(model, file_path):
    data = np.load(file_path)
    model.weights1 = data['weights1']
    model.weights2 = data['weights2']
    model.bias1 = data['bias1']
    model.bias2 = data['bias2']


def calculate_accuracy(y_true, y_pred):
    """
    Calculate the accuracy of the model.
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels.
    returns:
    float: Accuracy score
    """
    predictions = np.round(y_pred)
    correct_predictions = np.sum(predictions == y_true)
    return correct_predictions / len(y_true)