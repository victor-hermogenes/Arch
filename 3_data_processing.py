import numpy as np


def load_data(file_path):
    """
    Load data from a CSV file
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        np.ndarray: Loaded data as NumPy array.
    """
    return np.loadtxt(file_path, delimeter=",")


def normalize_data(data):
    """
    Normalize the data to be in the range [0, 1].
    Args:
        data (np.ndarray): Data to normalize.
    Returns:
        np.ndarray: Normalized data.
    """
    return data / np.max(data, axis=0)


def train_test_split(data, labels, test_size=0.2):
    """
    Split the data into training and testing sets.
    Args:
        data(np.ndarray): Feature data.
        labels (np.ndarray): Corresponding labels.
        test_size (float): Proportion of data to use for testing.
    Returns:
        tuple: Training data, testing data, training labels, testing labels
    """

    total_samples = data.shape[0]
    test_samples = int(total_samples * test_size)

    indices = np.random.permutation(total_samples)
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]


    return data[train_indices], data[test_indices], labels[train_indices], labels[test_indices]