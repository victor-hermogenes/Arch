import numpy as np
import pandas as pd


def load_data(file_path):
    """
    Load data from a CSV file
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        np.ndarray: Loaded data as NumPy array.
    """
    return pd.read_csv(file_path)


def process_data(df):
    """
    Process the data: encode text data as numbers.
    Args:
        df (pd.dataFrame): Data Frame cointaiining the data.
    Returns:
        np.ndarray: Feature data as a NumPy array.
        np.ndarray: Labels as a NumPy array.
    """
    df['question'] = df['question'].astype('category')
    df['answer'] = df['answer'].astype('category')
    X = df['Question'].cat.codes.values.reshape(-1, 1)
    y = df['Answer'].cat.codes.values.reshape(-1, 1)
    return X, y


def train_test_split(X, y, test_size=0.2):
    """
    Split the data into training and testing sets.
    Args:
        data(np.ndarray): Feature data.
        labels (np.ndarray): Corresponding labels.
        test_size (float): Proportion of data to use for testing.
    Returns:
        tuple: Training data, testing data, training labels, testing labels
    """

    total_samples = X.shape[0]
    test_samples = int(total_samples * test_size)

    indices = np.random.permutation(total_samples)
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]


    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]